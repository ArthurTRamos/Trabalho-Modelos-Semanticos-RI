"""Regressões do limite Top-100 exigido para a fusão RRF."""
import unittest

from ri_project.hybrid import HybridRRFRetriever, complete_ranking


class HybridTests(unittest.TestCase):
    def test_full_ranking_is_not_extended_or_scanned(self):
        ranking = [{'doc_id': f'doc-{i:03}', 'score': 100.0 - i} for i in range(100)]

        def forbidden_scan():
            raise AssertionError('Uma lista completa dispensa percorrer IDs da coleção.')
            yield

        self.assertEqual(complete_ranking(ranking, forbidden_scan(), 100), ranking)
        self.assertEqual(len(ranking), 100)
        self.assertEqual(complete_ranking(ranking, forbidden_scan(), 0), [])

    def test_short_ranking_gets_only_needed_zero_score_documents(self):
        ranking = [{'doc_id': 'B', 'score': 3.0}]
        self.assertEqual(complete_ranking(ranking, ['D', 'C', 'B', 'A'], 3), [
            {'doc_id': 'B', 'score': 3.0},
            {'doc_id': 'A', 'score': 0.0},
            {'doc_id': 'C', 'score': 0.0},
        ])

    def test_fusion_uses_only_top100_from_each_method(self):
        class Retriever:
            def __init__(self, prefix):
                self.prefix = prefix

            def search(self, query, top_k):
                return [{'doc_id': f'{self.prefix}-{i:03}', 'score': 1.0}
                        for i in range(top_k)]

        doc_ids = [f'{prefix}-{i:03}' for prefix in ('L', 'S', 'OUT') for i in range(150)]
        hybrid = HybridRRFRetriever(Retriever('L'), Retriever('S'),
                                    document_ids=doc_ids, k=10)
        result = hybrid.search('consulta', top_k=100)
        self.assertEqual(len(result), 100)
        self.assertTrue(all(not item['doc_id'].startswith('OUT-') for item in result))
        self.assertTrue(all(1 <= rank <= 100 for item in result
                            for rank in item['component_ranks'].values()))
        self.assertTrue(all(len(item['component_ranks']) == 1 for item in result))
        self.assertAlmostEqual(result[0]['score'], 1 / 11)


if __name__ == '__main__':
    unittest.main()
