"""Embeddings documentais e busca semântica Sentence-BERT."""
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import sentence_transformers
import torch
from sentence_transformers import SentenceTransformer

from .config import (EXPERIMENTS_DIR, SEMANTIC_BATCH_SIZE,
                     SEMANTIC_MODEL_NAME, SEMANTIC_NORMALIZE_EMBEDDINGS)


class SemanticRetriever:
    def __init__(self, documents, queries, qrels, model_name=SEMANTIC_MODEL_NAME,
                 cache_dir=None, batch_size=SEMANTIC_BATCH_SIZE,
                 normalize=SEMANTIC_NORMALIZE_EMBEDDINGS):
        self.qrels, self.queries = qrels, queries
        self.model_name, self.batch_size, self.normalize = model_name, batch_size, normalize
        self.cache_dir = Path(cache_dir or (EXPERIMENTS_DIR / 'semantic'))
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.embeddings_path = self.cache_dir / 'all_minilm_l6_v2_document_embeddings.npy'
        self.doc_ids_path = self.cache_dir / 'all_minilm_l6_v2_document_ids.json'
        self.manifest_path = self.cache_dir / 'all_minilm_l6_v2_manifest.json'
        self.document_ids = sorted(documents)
        self.document_texts = [
            f"{documents[d].get('title', '') or ''} {documents[d].get('text', '') or ''}".strip()
            for d in self.document_ids
        ]
        hasher = hashlib.sha256()
        for doc_id, text in zip(self.document_ids, self.document_texts):
            hasher.update(doc_id.encode()); hasher.update(b'\0')
            hasher.update(text.encode()); hasher.update(b'\n')
        self.corpus_sha256 = hasher.hexdigest()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = SentenceTransformer(model_name, device=self.device)
        self.max_seq_length = int(self.model.max_seq_length)
        token_lengths = []
        for start in range(0, len(self.document_texts), batch_size):
            batch = self.document_texts[start:start + batch_size]
            encoded = self.model.tokenizer(batch, add_special_tokens=True,
                truncation=False, padding=False)
            token_lengths.extend(len(ids) for ids in encoded['input_ids'])
        self.truncated_documents = sum(n > self.max_seq_length for n in token_lengths)
        self._load_or_encode()
        self.doc_id_to_row = {doc_id: row for row, doc_id in enumerate(self.document_ids)}

    def _load_or_encode(self):
        valid = False
        if all(p.exists() for p in (self.embeddings_path, self.doc_ids_path, self.manifest_path)):
            with self.manifest_path.open(encoding='utf-8') as f:
                manifest = json.load(f)
            with self.doc_ids_path.open(encoding='utf-8') as f:
                cached_ids = json.load(f)
            cached = np.load(self.embeddings_path, mmap_mode='r')
            valid = (manifest.get('model_name') == self.model_name and
                manifest.get('corpus_sha256') == self.corpus_sha256 and
                manifest.get('normalized_embeddings') is self.normalize and
                manifest.get('max_seq_length_tokens') == self.max_seq_length and
                cached_ids == self.document_ids and cached.ndim == 2 and
                cached.shape == (len(self.document_ids), self.model.get_embedding_dimension()))
        if valid:
            self.embeddings = np.load(self.embeddings_path)
            self.embedding_build_seconds = manifest.get('embedding_build_seconds')
            self.cache_status = 'carregado do cache validado'
            self.manifest = manifest
            return
        start = time.perf_counter()
        self.embeddings = self.model.encode(self.document_texts, batch_size=self.batch_size,
            show_progress_bar=True, convert_to_numpy=True,
            normalize_embeddings=self.normalize).astype(np.float32, copy=False)
        self.embedding_build_seconds = time.perf_counter() - start
        np.save(self.embeddings_path, self.embeddings)
        with self.doc_ids_path.open('w', encoding='utf-8') as f:
            json.dump(self.document_ids, f, ensure_ascii=False)
        self.manifest = {
            'model_name': self.model_name, 'corpus_sha256': self.corpus_sha256,
            'document_count': len(self.document_ids),
            'embedding_dimension': int(self.embeddings.shape[1]),
            'normalized_embeddings': self.normalize,
            'similarity': 'cosine (dot product of L2-normalized vectors)',
            'max_seq_length_tokens': self.max_seq_length,
            'truncated_document_count': int(self.truncated_documents),
            'embedding_build_seconds': self.embedding_build_seconds,
            'device': self.device, 'sentence_transformers_version': sentence_transformers.__version__,
            'torch_version': torch.__version__, 'embedding_dtype': str(self.embeddings.dtype),
        }
        with self.manifest_path.open('w', encoding='utf-8') as f:
            json.dump(self.manifest, f, ensure_ascii=False, indent=2)
        self.cache_status = 'gerado e salvo'

    def search(self, query, top_k=10):
        """Busca exaustiva por cosseno usando embeddings documentais em cache."""
        if top_k <= 0 or not self.document_ids:
            return []
        query_embedding = self.model.encode([str(query)], convert_to_numpy=True,
            normalize_embeddings=True)[0].astype(np.float32, copy=False)
        scores = self.embeddings @ query_embedding
        rows = np.argsort(-scores, kind='stable')[:top_k]
        return [{'doc_id': self.document_ids[row], 'score': float(scores[row])} for row in rows]

    def measure_query_time(self, split='dev', top_k=100, repetitions=1):
        if split not in self.qrels:
            raise ValueError(f'Partição desconhecida: {split}. Use train, dev ou test.')
        if repetitions <= 0:
            raise ValueError('repetitions deve ser maior que zero')
        query_ids = sorted(set(self.qrels[split]['query-id']) & set(self.queries))
        if not query_ids:
            return 0.0
        elapsed = 0.0
        for _ in range(repetitions):
            for qid in query_ids:
                start = time.perf_counter()
                self.search(self.queries[qid].get('text', ''), top_k)
                elapsed += time.perf_counter() - start
        return elapsed / (len(query_ids) * repetitions)
