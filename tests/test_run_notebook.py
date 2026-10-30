"""Limpeza de artefatos sem remover entradas ou seguir caches externos."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location(
    'notebook_runner', Path(__file__).resolve().parents[1] / 'scripts/run_notebook.py')
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class CleanNotebookTests(unittest.TestCase):
    def make_notebook(self, root):
        data = {'metadata': {'widgets': {'old': 'state'}}, 'cells': [
            {'cell_type': 'code', 'source': ['x = 1\n'], 'outputs': [{'text': 'old'}],
             'execution_count': 9, 'metadata': {'execution': {'old': 'time'}}}]}
        (root / 'Trabalho2.ipynb').write_text(json.dumps(data))
        return data

    def test_cleanup_preserves_dataset_environment_and_source(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            original = self.make_notebook(root)
            for name in ('nfcorpus/.cache/keep', '.venv/__pycache__/keep',
                         'ri_project/code.py', 'Experiments/cache/old',
                         'ri_project/__pycache__/old', '.ipynb_checkpoints/old'):
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text('contents')
            with patch.object(runner, 'ROOT', root):
                removed = runner.clean_generated_files()
            self.assertIn('Experiments/', removed)
            self.assertFalse((root / 'Experiments').exists())
            self.assertFalse((root / 'ri_project/__pycache__').exists())
            self.assertFalse((root / '.ipynb_checkpoints').exists())
            for name in ('nfcorpus/.cache/keep', '.venv/__pycache__/keep', 'ri_project/code.py'):
                self.assertEqual((root / name).read_text(), 'contents')
            notebook = json.loads((root / 'Trabalho2.ipynb').read_text())
            self.assertEqual(notebook['cells'][0]['source'], original['cells'][0]['source'])
            self.assertEqual(notebook['cells'][0]['outputs'], [])
            self.assertIsNone(notebook['cells'][0]['execution_count'])
            self.assertNotIn('widgets', notebook['metadata'])
            self.assertNotIn('execution', notebook['cells'][0]['metadata'])

    def test_nested_cache_link_does_not_delete_global_models(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / 'project'; root.mkdir()
            external = Path(directory) / 'shared'; external.mkdir()
            (external / 'weights').write_text('shared model')
            self.make_notebook(root)
            (root / 'Experiments/cache').mkdir(parents=True)
            (root / 'Experiments/cache/model').symlink_to(external, target_is_directory=True)
            with patch.object(runner, 'ROOT', root):
                runner.clean_generated_files()
            self.assertEqual((external / 'weights').read_text(), 'shared model')
            self.assertFalse((root / 'Experiments').exists())

    def test_experiments_directory_link_is_only_unlinked(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / 'project'; root.mkdir()
            external = Path(directory) / 'shared'; external.mkdir()
            (external / 'weights').write_text('shared model')
            self.make_notebook(root)
            (root / 'Experiments').symlink_to(external, target_is_directory=True)
            with patch.object(runner, 'ROOT', root):
                runner.clean_generated_files()
            self.assertEqual((external / 'weights').read_text(), 'shared model')
            self.assertFalse((root / 'Experiments').is_symlink())


if __name__ == '__main__':
    unittest.main()
