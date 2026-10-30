"""Contratos do experimento: sorteio, isolamento de A/B e retomada segura."""
from contextlib import redirect_stdout
import csv
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest

ROOT = Path(__file__).resolve().parents[1]
package = types.ModuleType('experiment_test_package')
package.__path__ = [str(ROOT / 'ri_project')]
sys.modules[package.__name__] = package
for name in ('rag', 'rag_experiment'):
    module_name = f'{package.__name__}.{name}'
    spec = importlib.util.spec_from_file_location(module_name, ROOT / 'ri_project' / f'{name}.py')
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
experiment = sys.modules[f'{package.__name__}.rag_experiment']
rag = sys.modules[f'{package.__name__}.rag']


class Generator:
    def __init__(self):
        self.config = rag.RAGConfig()
        self.calls = []

    def manifest(self):
        return {'model': 'test-double', 'revision': 'fixed'}

    def generate(self, messages):
        self.calls.append(messages)
        return {'answer': 'Test answer.', 'prompt': json.dumps(messages),
                'input_tokens': 5, 'output_tokens': 3,
                'generation_seconds': 0.01}


def make_bundle():
    return {'query_ids': list(experiment.RAG_QUERY_IDS), 'cases': [
        {'query_id': qid, 'query': f'Question for {qid}?', 'contexts': [
            {'doc_id': f'D{i}', 'title': f'Title {i}', 'text': f'EVIDENCE {i}'}
            for i in range(5)]} for qid in experiment.RAG_QUERY_IDS]}


class ExperimentTests(unittest.TestCase):
    @unittest.skipUnless((ROOT / 'nfcorpus/nfcorpus/queries.jsonl').exists(), 'Dataset opcional ausente')
    def test_official_dataset_sampling(self):
        data = ROOT / 'nfcorpus/nfcorpus'
        queries = {r['_id']: r for r in map(json.loads, (data / 'queries.jsonl').read_text().splitlines())}
        with (data / 'qrels/test.tsv').open() as file:
            ids = [r['query-id'] for r in csv.DictReader(file, delimiter='\t')]
        self.assertEqual(experiment.select_experiment_queries(queries, ids), list(experiment.RAG_QUERY_IDS))

    def test_twenty_paired_answers_have_identical_generator_and_no_context_leak(self):
        generator = Generator()
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            rows = experiment.run_paired_experiment(make_bundle(), generator, directory)
            status = json.loads((Path(directory) / 'run_status.json').read_text())
            self.assertEqual(status['status'], 'complete')
            self.assertEqual(status['completed_responses'], 20)
            self.assertEqual(len(rows), 20)
            with (Path(directory) / 'rag_experiment.csv').open() as file:
                exported = list(csv.DictReader(file))
            self.assertEqual(len(exported), 20)
            self.assertEqual(exported[0]['condition'], 'A')
            self.assertEqual(exported[0]['context_doc_ids'], '[]')
            self.assertEqual(exported[0]['rastreabilidade'], 'N/A')
            self.assertEqual(exported[0]['fidelidade'], '')
            self.assertEqual(exported[1]['condition'], 'B')
            self.assertEqual(len(json.loads(exported[1]['context_doc_ids'])), 5)
            self.assertEqual(exported[1]['rastreabilidade'], '')
            for a, b in zip(rows[::2], rows[1::2]):
                self.assertEqual(a['manifest'], b['manifest'])
                self.assertEqual(a['query'], b['query'])
                self.assertEqual(a['contexts'], [])
                self.assertFalse(a['external_context_provided'])
                self.assertEqual(a['messages'][1]['content'], a['query'])
                self.assertNotIn('EVIDENCE', json.dumps(a['messages']))
                self.assertIn('EVIDENCE', json.dumps(b['messages']))
                self.assertEqual(len(b['contexts']), 5)
            experiment.run_paired_experiment(make_bundle(), generator, directory)
            self.assertEqual(len(generator.calls), 20, 'Retomada idêntica deve reutilizar checkpoints.')
            changed = make_bundle()
            changed['cases'][0]['contexts'][0]['text'] = 'Different evidence'
            experiment.run_paired_experiment(changed, generator, directory)
            self.assertEqual(len(generator.calls), 40, 'Contexto alterado invalida a execução anterior.')

    def test_remeasured_timings_do_not_invalidate_responses(self):
        generator = Generator()
        bundle = make_bundle()
        bundle['cases'][0]['retrieval_and_reranking_seconds'] = 1.0
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            experiment.run_paired_experiment(bundle, generator, directory)
            bundle['cases'][0]['retrieval_and_reranking_seconds'] = 2.0
            experiment.run_paired_experiment(bundle, generator, directory)
        self.assertEqual(len(generator.calls), 20)

    def test_protocol_mismatch_rejected_before_generation(self):
        generator = Generator()
        with tempfile.TemporaryDirectory() as directory:
            bundle = make_bundle()
            bundle['cases'] = list(reversed(bundle['cases']))
            with self.assertRaises(ValueError):
                experiment.run_paired_experiment(bundle, generator, directory)
            self.assertEqual(generator.calls, [])

    def test_export_preserves_human_notes_only_for_unchanged_answers(self):
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            rows = experiment.run_paired_experiment(make_bundle(), Generator(), directory)
            path = Path(directory) / 'rag_experiment.csv'
            with path.open() as file:
                reader = csv.DictReader(file)
                saved, fields = list(reader), reader.fieldnames
            saved[0]['fidelidade'] = '2'
            saved[0]['justificativa_fidelidade'] = 'Synthetic human annotation.'
            with path.open('w', newline='') as file:
                writer = csv.DictWriter(file, fieldnames=fields)
                writer.writeheader()
                writer.writerows(saved)
            experiment.export_standardized_csv(rows, path)
            with path.open() as file:
                preserved = list(csv.DictReader(file))
            self.assertEqual(preserved[0]['fidelidade'], '2')
            changed = [dict(row) for row in rows]
            changed[0]['answer'] = 'Changed answer.'
            experiment.export_standardized_csv(changed, path)
            with path.open() as file:
                invalidated = list(csv.DictReader(file))
            self.assertEqual(invalidated[0]['fidelidade'], '')
            self.assertEqual(invalidated[0]['justificativa_fidelidade'], '')

    def test_interrupted_run_is_incomplete_and_resumes_saved_answer(self):
        class FailOnce(Generator):
            failed = False

            def generate(self, messages):
                if len(self.calls) == 1 and not self.failed:
                    self.failed = True
                    raise RuntimeError('Interrupção simulada')
                return super().generate(messages)

        generator = FailOnce()
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(RuntimeError, 'Interrupção'):
                experiment.run_paired_experiment(make_bundle(), generator, directory)
            status = json.loads((Path(directory) / 'run_status.json').read_text())
            self.assertEqual(status['status'], 'incomplete')
            self.assertEqual(status['completed_responses'], 1)
            rows = experiment.run_paired_experiment(make_bundle(), generator, directory)
            self.assertEqual(len(rows), 20)
            self.assertEqual(len(generator.calls), 20)


if __name__ == '__main__':
    unittest.main()
