"""Contratos de métricas, limites de candidatos e cache sem downloads de modelos."""
import math
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from ri_project.bm25 import BM25Retriever
from ri_project.evaluation import retrieval_metrics
from ri_project.hybrid import HybridRRFRetriever
from ri_project.indexing import build_inverted_index
from ri_project.reranking import HybridRerankingRetriever
from ri_project.semantic import SemanticRetriever
from ri_project.typo import character_kgrams, edit_distance, jaccard_coefficient


class RetrievalContractsTests(unittest.TestCase):
    def test_metrics_use_all_qrels_and_linear_gain(self):
        scores = retrieval_metrics(['b', 'x', 'a'], {'a': 2, 'b': 1, 'c': 1}, k=3)
        self.assertAlmostEqual(scores['P10'], 2 / 3)
        self.assertAlmostEqual(scores['R10'], 2 / 3)
        self.assertAlmostEqual(scores['AP10'], (1 + 2 / 3) / 3)
        self.assertAlmostEqual(scores['NDCG10'], 2 / (2 + 1 / math.log2(3) + .5))
        with self.assertRaises(ValueError):
            retrieval_metrics(['a', 'a'], {'a': 1})

    def test_index_and_bm25_use_postings_only(self):
        documents = {'A': {'title': 'apple', 'text': 'apple banana'},
                     'B': {'title': 'banana', 'text': 'pear'}}
        index = build_inverted_index(documents)
        self.assertEqual(index['num_postings'], 4)
        self.assertEqual(index['inverted_index']['appl'], {'A': 2})
        self.assertEqual(index['document_lengths'], {'A': 3, 'B': 2})
        retriever = BM25Retriever(index)
        class LookupOnly(dict):
            def __iter__(self):
                raise AssertionError('Busca não deve varrer documentos.')
            def items(self):
                raise AssertionError('Busca não deve varrer documentos.')
        retriever.document_lengths = LookupOnly(index['document_lengths'])
        ranking = retriever.search('apple')
        self.assertEqual([r['doc_id'] for r in ranking], ['A'])
        expected = math.log(2) * (2 * 2.2) / (2 + 1.2 * (.25 + .75 * 3 / 2.5))
        self.assertAlmostEqual(ranking[0]['score'], expected)

    def test_hybrid_output_depth_does_not_expand_candidates(self):
        class Retriever:
            def __init__(self, prefix):
                self.prefix = prefix
                self.depths = []
            def search(self, query, top_k):
                self.depths.append(top_k)
                return [{'doc_id': f'{self.prefix}{i}', 'score': 1} for i in range(top_k)]
        left, right = Retriever('L'), Retriever('S')
        hybrid = HybridRRFRetriever(left, right, document_ids=['unused'])
        ranking = hybrid.search('q', top_k=150)
        self.assertEqual(left.depths, [100])
        self.assertEqual(right.depths, [100])
        self.assertEqual(len(ranking), 150)
        self.assertTrue(all(rank <= 100 for item in ranking for rank in item['component_ranks'].values()))

    def test_top5_reranks_all_fifty_candidates(self):
        class Hybrid:
            def search(self, query, top_k):
                return [{'doc_id': str(i), 'score': i} for i in range(top_k)]
        class Reranker:
            rerank_depth = 50
            def rerank(self, query, ranking, documents):
                self.count = len(ranking)
                return list(reversed(ranking))
        reranker = Reranker()
        wrapper = HybridRerankingRetriever(Hybrid(), reranker, {})
        self.assertEqual(wrapper.search('q', 5)[0]['doc_id'], '49')
        self.assertEqual(reranker.count, 50)
        self.assertEqual(wrapper.search('q', 0), [])

    def test_spelling_primitives(self):
        self.assertEqual(edit_distance('kitten', 'sitting'), 3)
        self.assertEqual(edit_distance('ab', 'ba'), 2)
        self.assertEqual(character_kgrams('cat', 2), {'$c', 'ca', 'at', 't$'})
        self.assertAlmostEqual(jaccard_coefficient({'a', 'b'}, {'b', 'c'}), 1 / 3)

    def test_semantic_cache_invalidates_changed_revision(self):
        class Model:
            max_seq_length = 256
            encodes = 0
            def __init__(self, *args, **kwargs):
                self.tokenizer = lambda batch, **kw: {'input_ids': [[1, 2] for _ in batch]}
            def get_embedding_dimension(self):
                return 2
            def encode(self, texts, **kwargs):
                Model.encodes += 1
                return np.array([[1., 0.] for _ in texts])
        docs = {'A': {'title': 'title', 'text': 'text'}}
        with tempfile.TemporaryDirectory() as directory, patch('ri_project.semantic.SentenceTransformer', Model):
            for revision in ('first', 'first', 'second'):
                retriever = SemanticRetriever(docs, {}, {}, cache_dir=directory, revision=revision)
                self.assertEqual(retriever.manifest['model_revision'], revision)
            self.assertEqual(Model.encodes, 2)
            with self.assertRaises(ValueError):
                SemanticRetriever(docs, {}, {}, cache_dir=directory, normalize=False)
