"""Exercise generated submission setup from an unrelated caller directory.

The sbatch executable is a local capture fixture: no real jobs are submitted.
The test itself must execute inside a genuine Slurm compute allocation.
"""
import os
from pathlib import Path
import shlex
import subprocess
import tempfile
import unittest

from comparison.study import submission_header


@unittest.skipUnless(os.environ.get('SLURMD_NODENAME') and os.environ.get('SLURM_JOB_ID'),
                     'Submission regression must run inside a real Slurm allocation')
class SubmissionDirectoryTests(unittest.TestCase):
    def test_sbatch_inherits_snapshot_even_from_wrong_directory(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            # Spaces and apostrophes also exercise actual shell path quoting.
            study = root / "study's directory"
            snapshot = study / 'snapshot'
            snapshot.mkdir(parents=True)
            wrong = root / 'unrelated-caller'
            wrong.mkdir()
            binary = root / 'fake-bin'
            binary.mkdir()
            capture = root / 'capture.txt'
            stub = binary / 'sbatch'
            stub.write_text('#!/usr/bin/env bash\nset -euo pipefail\npwd >> "$CAPTURE"\nprintf "fixture-job-id\\n"\n')
            stub.chmod(0o755)
            # Both generators consume this shared header. Include both actual
            # submission forms to check --chdir does not substitute for cd.
            for kind in ('study', 'tools'):
                with self.subTest(kind=kind):
                    script = study / (kind + '-submit.sh')
                    script.write_text('\n'.join(submission_header(study) + [
                        'sbatch --parsable --chdir ' + shlex.quote(str(snapshot)) + ' fixture.sbatch']) + '\n')
                    environment = {**os.environ, 'PATH': str(binary) + os.pathsep + os.environ['PATH'],
                                   'CAPTURE': str(capture)}
                    result = subprocess.run(['bash', str(script)], cwd=wrong, env=environment,
                                            text=True, capture_output=True, check=True)
                    self.assertEqual(result.stdout.strip(), 'fixture-job-id')
            self.assertEqual(capture.read_text().splitlines(), [str(snapshot), str(snapshot)])

    def test_missing_snapshot_fails_before_submission(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            script = root / 'submit.sh'
            script.write_text('\n'.join(submission_header(root) + ['exit 17']) + '\n')
            result = subprocess.run(['bash', str(script)], cwd='/', text=True, capture_output=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertNotEqual(result.returncode, 17)

    def test_nested_submission_does_not_forward_parent_cpu_mask(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / 'snapshot').mkdir()
            script = root / 'submit.sh'
            script.write_text('\n'.join(submission_header(root) + [
                'env']) + '\n')
            environment = {**os.environ, 'SLURM_CPU_BIND': 'mask_cpu:0xff00',
                           'SLURM_CPU_BIND_LIST': '0xff00', 'SLURM_CPU_BIND_TYPE': 'mask_cpu',
                           'SLURM_CPU_BIND_VERBOSE': 'verbose', 'SRUN_CPU_BIND': 'cores'}
            result = subprocess.run(['bash', str(script)], env=environment,
                                    text=True, capture_output=True, check=True)
            actual = dict(line.split('=', 1) for line in result.stdout.splitlines() if '=' in line)
            self.assertEqual(actual['SLURM_CPU_BIND'], 'none')
            self.assertEqual(actual['SRUN_CPU_BIND'], 'none')
            for name in ('SLURM_CPU_BIND_LIST', 'SLURM_CPU_BIND_TYPE', 'SLURM_CPU_BIND_VERBOSE'):
                self.assertNotIn(name, actual)


if __name__ == '__main__':
    unittest.main()
