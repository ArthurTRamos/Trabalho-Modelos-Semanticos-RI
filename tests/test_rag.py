"""Verificação offline dos contratos do RAG, sem baixar pesos ou dataset."""
import importlib.util
from contextlib import nullcontext
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest


# Carrega o módulo isoladamente: __init__ do pacote exige NLTK/pandas.
spec = importlib.util.spec_from_file_location(
    'rag_under_test', Path(__file__).resolve().parents[1] / 'ri_project' / 'rag.py',
)
rag = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = rag
spec.loader.exec_module(rag)


class FakeGenerator:
    def __init__(self, **kwargs):
        self.config = rag.RAGConfig(**kwargs)

    def generate(self, messages):
        self.messages = messages
        return {'answer': 'Evidence [D9].', 'input_tokens': 100,
                'output_tokens': 5, 'generation_seconds': 0.01}

    def manifest(self):
        return {'execution': 'test double'}


class FakeRetriever:
    def __init__(self):
        self.reranker = SimpleNamespace(rerank_depth=50, manifest=lambda: {'depth': 50})
        self.hybrid_retriever = SimpleNamespace(k=10, candidate_depth=100, weights={})

    def search(self, query, top_k):
        self.requested_depth = top_k
        return [{'doc_id': f'D{i}', 'score': float(i), 'hybrid_rank': 10 - i}
                for i in range(9, -1, -1)]


class RAGTests(unittest.TestCase):
    def setUp(self):
        self.documents = {f'D{i}': {'title': f'Title {i}', 'text': f'Full text {i}.'}
                          for i in range(10)}

    def test_reranking_depth_context_order_and_audit_file(self):
        retriever = FakeRetriever()
        generator = FakeGenerator()
        pipeline = rag.RAGPipeline(retriever, self.documents, generator)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'nested' / 'answer.json'
            result = pipeline.answer('Question?', 'Q1', output_path=path)
            saved = json.loads(path.read_text())
        self.assertEqual(retriever.requested_depth, 100)
        self.assertEqual([c['doc_id'] for c in saved['contexts']],
                         ['D9', 'D8', 'D7', 'D6', 'D5'])
        self.assertEqual(saved['contexts'][0]['text'], 'Full text 9.')
        self.assertEqual(saved['contexts'][0]['title'], 'Title 9')
        self.assertEqual(saved['answer'], result['answer'])
        self.assertEqual(saved['query_id'], 'Q1')
        self.assertEqual(saved['messages'], generator.messages)
        self.assertFalse(saved['retrieval']['truncation'])

    def test_prompt_preserves_sources_and_requires_evidence(self):
        self.documents['D1']['text'] = 'Text with "quotes" and\nnewlines.'
        contexts = rag.select_context([{'doc_id': 'D1', 'score': 1}], self.documents, 1)
        messages = rag.build_rag_messages('Question?', contexts)
        content = messages[1]['content'].split('Documents (JSON):\n')[1]
        sources = json.loads(content.split('\n\nQuestion:')[0])
        self.assertEqual(sources[0]['text'], self.documents['D1']['text'])
        self.assertIn('doc_id', messages[0]['content'])
        self.assertIn('sufficient evidence', messages[0]['content'])
        self.assertIn('English', messages[0]['content'])

    def test_invalid_context_fails_instead_of_substituting_sources(self):
        with self.assertRaises(ValueError):
            rag.select_context([], self.documents)
        with self.assertRaises(ValueError):
            rag.select_context([{'doc_id': 'D1', 'score': 1}] * 5, self.documents)
        with self.assertRaises(KeyError):
            rag.select_context([{'doc_id': 'missing', 'score': 1}], self.documents, 1)

    def test_invalid_configuration_and_empty_question(self):
        for kwargs in ({'temperature': -1}, {'max_new_tokens': 0},
                       {'context_documents': 0}, {'language': ''}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                rag.RAGConfig(**kwargs)
        with self.assertRaises(ValueError):
            rag.RAGPipeline(FakeRetriever(), self.documents,
                            FakeGenerator(retrieval_depth=5))
        pipeline = rag.RAGPipeline(FakeRetriever(), self.documents, FakeGenerator())
        with self.assertRaises(ValueError):
            pipeline.answer(' ')

    def test_overflow_fails_before_generation(self):
        class FakeInputs(dict):
            def to(self, device):
                return self

        class FakeTokenizer:
            def apply_chat_template(self, *args, **kwargs):
                return 'full prompt'

            def __call__(self, prompt, **kwargs):
                self.options = kwargs
                return FakeInputs(input_ids=SimpleNamespace(shape=(1, 100)))

        generator = object.__new__(rag.LocalCausalGenerator)
        generator.config = rag.RAGConfig(max_new_tokens=20)
        generator.device = 'cpu'
        generator.torch = None
        generator.tokenizer = FakeTokenizer()
        generator.context_window = 119
        with self.assertRaisesRegex(ValueError, 'nenhum documento foi truncado'):
            generator.generate([])
        self.assertFalse(generator.tokenizer.options['truncation'])

    def test_local_adapter_decodes_only_answer_and_controls_sampling(self):
        class FakeInputs(dict):
            def to(self, device):
                return self

        class FakeOutput:
            def __getitem__(self, index):
                # Exige que somente os tokens após os 100 do prompt sejam lidos.
                if index != (0, slice(100, None)):
                    raise AssertionError(index)
                return [201, 202]

        class FakeTokenizer:
            eos_token_id = 0

            def apply_chat_template(self, messages, **kwargs):
                return 'formatted full prompt'

            def __call__(self, prompt, **kwargs):
                return FakeInputs(input_ids=SimpleNamespace(shape=(1, 100)))

            def decode(self, tokens, **kwargs):
                self.decoded = tokens
                return ' Answer [D9]. '

        class FakeModel:
            def generate(self, **kwargs):
                self.kwargs = kwargs
                return FakeOutput()

        for temperature in (0, 0.7):
            with self.subTest(temperature=temperature):
                generator = object.__new__(rag.LocalCausalGenerator)
                generator.config = rag.RAGConfig(temperature=temperature, max_new_tokens=20)
                generator.device = 'cpu'
                seeds = []
                generator.torch = SimpleNamespace(manual_seed=seeds.append,
                                                 inference_mode=nullcontext)
                generator.tokenizer = FakeTokenizer()
                generator.model = FakeModel()
                generator.context_window = 120
                result = generator.generate([{'role': 'user', 'content': 'Question?'}])
                self.assertEqual(result['answer'], 'Answer [D9].')
                self.assertEqual(result['input_tokens'], 100)
                self.assertEqual(result['output_tokens'], 2)
                self.assertEqual(result['prompt'], 'formatted full prompt')
                self.assertEqual(seeds, [42])
                self.assertEqual(generator.model.kwargs['do_sample'], temperature > 0)
                self.assertEqual(generator.model.kwargs['max_new_tokens'], 20)
                if temperature > 0:
                    self.assertEqual(generator.model.kwargs['temperature'], temperature)


if __name__ == '__main__':
    unittest.main()
