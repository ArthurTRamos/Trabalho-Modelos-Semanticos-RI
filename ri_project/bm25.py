"""Busca lexical BM25 sobre o índice invertido."""
import math
from collections import defaultdict
from time import perf_counter

from .config import BM25_B, BM25_K1
from .preprocessing import preprocess


class BM25Retriever:
    def __init__(self, index, k1=BM25_K1, b=BM25_B):
        self.inverted_index = index['inverted_index']
        self.document_lengths = index['document_lengths']
        self.num_documents = index['num_documents']
        self.avg_document_length = index['avg_document_length']
        self.k1, self.b = k1, b

    def search(self, query, top_k=10, k1=None, b=None):
        """Retorna os documentos mais bem pontuados por BM25."""
        k1 = self.k1 if k1 is None else k1
        b = self.b if b is None else b
        if k1 <= 0 or not 0 <= b <= 1:
            raise ValueError('BM25 requer k1 > 0 e 0 <= b <= 1')
        if top_k <= 0 or self.num_documents == 0 or self.avg_document_length == 0:
            return []

        scores = defaultdict(float)
        for term in set(preprocess(query)):
            postings = self.inverted_index.get(term)
            if not postings:
                continue
            df = len(postings)
            idf = math.log1p((self.num_documents - df + 0.5) / (df + 0.5))
            for doc_id, tf in postings.items():
                length = self.document_lengths[doc_id]
                norm = k1 * (1 - b + b * length / self.avg_document_length)
                scores[doc_id] += idf * (tf * (k1 + 1)) / (tf + norm)

        ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))[:top_k]
        return [{'doc_id': doc_id, 'score': float(score)} for doc_id, score in ranked]

    def measure_query_time(self, queries, qrels, split='dev', top_k=100, repetitions=1):
        """Mede a latência média de busca para as consultas julgadas."""
        if split not in qrels:
            raise ValueError(f'Partição desconhecida: {split}. Use train, dev ou test.')
        if repetitions <= 0:
            raise ValueError('repetitions deve ser maior que zero')
        query_ids = sorted(set(qrels[split]['query-id']) & set(queries))
        if not query_ids:
            raise ValueError(f'Nenhuma consulta de {split} foi encontrada em queries.jsonl')
        start = perf_counter()
        for _ in range(repetitions):
            for query_id in query_ids:
                self.search(queries[query_id].get('text', ''), top_k=top_k)
        return {
            'split': split,
            'num_queries': len(query_ids),
            'repetitions': repetitions,
            'average_query_time_seconds': (perf_counter() - start) / (len(query_ids) * repetitions),
        }
