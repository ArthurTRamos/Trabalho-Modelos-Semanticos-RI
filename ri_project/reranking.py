"""Reranking do Top-50 híbrido com um cross-encoder pré-treinado."""
import json
import time
from pathlib import Path

import numpy as np
import sentence_transformers
import torch
from sentence_transformers import CrossEncoder

RERANKER_MODEL_NAME = 'cross-encoder/ms-marco-MiniLM-L6-v2'
RERANKER_MODEL_REVISION = '233902d25c440f23af6f7d6e94d2946bac0bee0a'
RERANK_DEPTH = 50
RERANK_BATCH_SIZE = 32


class CrossEncoderReranker:
    """Reordena os primeiros resultados com pontuação conjunta consulta-documento."""
    def __init__(self, model_name=RERANKER_MODEL_NAME, rerank_depth=RERANK_DEPTH,
                 batch_size=RERANK_BATCH_SIZE, device=None,
                 revision=RERANKER_MODEL_REVISION):
        if rerank_depth <= 0 or batch_size <= 0:
            raise ValueError('rerank_depth e batch_size devem ser positivos.')
        self.model_name = model_name
        self.revision = revision
        self.rerank_depth = rerank_depth
        self.batch_size = batch_size
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = CrossEncoder(model_name, revision=revision, device=self.device)
        configured_max = getattr(self.model, 'max_length', None)
        tokenizer_max = getattr(self.model.tokenizer, 'model_max_length', None)
        self.max_input_tokens = int(configured_max or tokenizer_max or 512)
        self.last_reranking_seconds = 0.0
        self.last_truncated_pairs = 0
        self.total_pairs = 0
        self.total_truncated_pairs = 0
        self.max_observed_pair_tokens = 0

    @staticmethod
    def _document_text(document):
        title = str(document.get('title', '') or '')
        text = str(document.get('text', '') or '')
        return f'{title} {text}'.strip()

    def rerank(self, query, ranking, documents):
        """Reordena Top-N e mantém os itens abaixo de N na ordem original."""
        start = time.perf_counter()
        head = list(ranking[:self.rerank_depth])
        tail = list(ranking[self.rerank_depth:])
        if not head:
            self.last_reranking_seconds = time.perf_counter() - start
            self.last_truncated_pairs = 0
            return []

        document_texts = [self._document_text(documents[item['doc_id']]) for item in head]
        queries = [str(query)] * len(head)
        encoded = self.model.tokenizer(
            queries, document_texts, add_special_tokens=True,
            truncation=False, padding=False,
        )
        token_lengths = [len(ids) for ids in encoded['input_ids']]
        self.last_truncated_pairs = sum(length > self.max_input_tokens for length in token_lengths)
        self.max_observed_pair_tokens = max(self.max_observed_pair_tokens, max(token_lengths, default=0))
        self.total_pairs += len(head)
        self.total_truncated_pairs += self.last_truncated_pairs

        pairs = [(str(query), text) for text in document_texts]
        scores = np.asarray(self.model.predict(
            pairs, batch_size=self.batch_size, show_progress_bar=False,
            convert_to_numpy=True,
        )).reshape(-1)
        reranked_head = []
        for original_rank, (item, score) in enumerate(zip(head, scores), start=1):
            reranked_head.append({
                'doc_id': item['doc_id'],
                'score': float(score),
                'reranker_score': float(score),
                'hybrid_score': float(item['score']),
                'hybrid_rank': original_rank,
            })
        reranked_head.sort(key=lambda item: (-item['score'], item['hybrid_rank'], item['doc_id']))

        # Mantém a cauda com a ordem e os scores híbridos originais.
        preserved_tail = []
        for offset, item in enumerate(tail, start=self.rerank_depth + 1):
            preserved_tail.append({
                'doc_id': item['doc_id'], 'score': float(item['score']),
                'reranker_score': None, 'hybrid_score': float(item['score']),
                'hybrid_rank': offset,
            })
        self.last_reranking_seconds = time.perf_counter() - start
        return reranked_head + preserved_tail

    def reset_statistics(self):
        """Zera contadores para registrar uma partição isoladamente."""
        self.total_pairs = 0
        self.total_truncated_pairs = 0
        self.max_observed_pair_tokens = 0

    def manifest(self):
        return {
            'model_name': self.model_name,
            'model_revision': self.revision,
            'sentence_transformers_version': sentence_transformers.__version__,
            'torch_version': str(torch.__version__),
            'rerank_depth': self.rerank_depth,
            'batch_size': self.batch_size,
            'device': self.device,
            'max_input_tokens': self.max_input_tokens,
            'total_pairs_scored': self.total_pairs,
            'truncated_pair_count': self.total_truncated_pairs,
            'max_untruncated_pair_length_observed': self.max_observed_pair_tokens,
        }

    def save_manifest(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.manifest(), ensure_ascii=False, indent=2), encoding='utf-8')


class HybridRerankingRetriever:
    """Recupera com RRF e aplica o cross-encoder somente ao Top-N."""
    def __init__(self, hybrid_retriever, reranker, documents):
        self.hybrid_retriever = hybrid_retriever
        self.reranker = reranker
        self.documents = documents
        self.last_hybrid_retrieval_seconds = 0.0

    def search(self, query, top_k=100):
        """Reranqueia toda a cabeça configurada antes de cortar a saída."""
        if top_k <= 0:
            self.last_hybrid_retrieval_seconds = 0.0
            self.reranker.last_reranking_seconds = 0.0
            return []
        start = time.perf_counter()
        depth = max(top_k, self.reranker.rerank_depth)
        ranking = self.hybrid_retriever.search(query, top_k=depth)
        self.last_hybrid_retrieval_seconds = time.perf_counter() - start
        return self.reranker.rerank(query, ranking, self.documents)[:top_k]
