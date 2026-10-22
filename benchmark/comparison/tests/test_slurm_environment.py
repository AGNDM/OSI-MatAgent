"""Check allocation and inherited-mask handling without submitting real jobs."""
import os
from pathlib import Path
import subprocess
import unittest


@unittest.skipUnless(os.environ.get('SLURMD_NODENAME') and os.environ.get('SLURM_JOB_ID'),
                     'Launcher checks must run inside a real Slurm allocation')
class SlurmEnvironmentTests(unittest.TestCase):
    helper = Path(__file__).resolve().parents[1] / 'slurm/common.sh'

    def test_nested_allocation_discards_parent_mask_and_uses_allocated_threads(self):
        environment = {**os.environ, 'SLURM_CPU_BIND': 'mask_cpu:0xff00',
                       'SLURM_CPU_BIND_LIST': '0xff00', 'SLURM_CPU_BIND_TYPE': 'mask_cpu',
                       'SLURM_CPU_BIND_VERBOSE': 'verbose', 'SRUN_CPU_BIND': 'cores',
                       'SRUN_CPU_BIND_LIST': '8-15', 'SRUN_CPU_BIND_TYPE': 'cores',
                       'SRUN_CPU_BIND_VERBOSE': 'verbose', 'SLURM_CPUS_PER_TASK': '3',
                       'PYTHONPATH': '/unrelated/site-packages', 'PYTHONHOME': '/unrelated/python'}
        result = subprocess.run(
            ['bash', '-c', 'set -euo pipefail; source "$1"; comparison_inference_environment; env',
             'comparison-environment-test', str(self.helper)],
            env=environment, text=True, capture_output=True, check=True)
        actual = dict(line.split('=', 1) for line in result.stdout.splitlines() if '=' in line)
        self.assertEqual(actual['SLURM_CPU_BIND'], 'none')
        self.assertEqual(actual['SRUN_CPU_BIND'], 'none')
        for prefix in ('SLURM_CPU_BIND', 'SRUN_CPU_BIND'):
            for suffix in ('LIST', 'TYPE', 'VERBOSE'):
                self.assertNotIn(prefix + '_' + suffix, actual)
        for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
            self.assertEqual(actual[name], '3')
        self.assertNotIn('PYTHONPATH', actual)
        self.assertNotIn('PYTHONHOME', actual)
        self.assertEqual(actual['HF_HUB_OFFLINE'], '1')

    def test_missing_allocation_fails_before_environment_setup(self):
        environment = {key: value for key, value in os.environ.items()
                       if key not in {'SLURMD_NODENAME', 'SLURM_JOB_ID'}}
        result = subprocess.run(
            ['bash', '-c', 'set -euo pipefail; source "$1"; printf reached',
             'comparison-environment-test', str(self.helper)],
            env=environment, text=True, capture_output=True)
        self.assertEqual(result.returncode, 2)
        self.assertNotIn('reached', result.stdout)
        self.assertIn('real Slurm compute allocation', result.stderr)


if __name__ == '__main__':
    unittest.main()
