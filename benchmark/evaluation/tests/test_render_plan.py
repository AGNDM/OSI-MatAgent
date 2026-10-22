import importlib.util
import os
from pathlib import Path
import unittest

if not os.environ.get('SLURM_JOB_ID') or not os.environ.get('SLURMD_NODENAME'):
    raise RuntimeError('A genuine Slurm compute allocation is required')
spec = importlib.util.spec_from_file_location('evaluation_plan_renderer',
    Path(__file__).resolve().parents[1] / 'workflow/render_plan.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class ExplicitBindings(unittest.TestCase):
    def test_exact_paths_resolve_without_changing_model_profile_or_arguments(self):
        original = {'expert': {'model_path': '${CHECKPOINT}', 'dtype': 'float16',
                              'max_context': 65536}, 'arguments': {'temperature_k': 303.15}}
        result = module.render(original, {'CHECKPOINT': '/models/explicit'})
        self.assertEqual(result['expert']['model_path'], '/models/explicit')
        self.assertEqual(result['expert']['max_context'], 65536)
        self.assertEqual(result['arguments'], original['arguments'])
        self.assertEqual(original['expert']['model_path'], '${CHECKPOINT}')

    def test_missing_relative_or_partial_bindings_are_rejected(self):
        with self.assertRaises(KeyError):
            module.render('${CHECKPOINT}', {})
        for value in ('relative/path', '${UNRESOLVED}'):
            with self.assertRaises(ValueError):
                module.render('${CHECKPOINT}', {'CHECKPOINT': value})
        with self.assertRaises(ValueError):
            module.render('/models/${CHECKPOINT}', {'CHECKPOINT': '/models/explicit'})


if __name__ == '__main__':
    unittest.main(verbosity=2)
