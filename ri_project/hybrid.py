"""Fusão lexical-semântica por Reciprocal Rank Fusion (RRF)."""
from collections import defaultdict

RRF_K = 60
HYBRID_CANDIDATE_DEPTH = 100


def complete_ranking(ranking, document_ids, depth):
    """Completa rankings curtos com documentos sem score em ordem de ID."""
    completed = list(ranking[:depth])
    if len(completed) == depth:
        return completed
    present = {item['doc_id'] for item in completed}
    for doc_id in sorted(document_ids):
        if doc_id not in present:
            completed.append({'doc_id': doc_id, 'score': 0.0})
            present.add(doc_id)
            if len(completed) == depth:
                break
    return completed


def reciprocal_rank_fusion(rankings, k=RRF_K, weights=None, top_k=None):
    """Funde rankings ordenados usando ``peso / (k + rank)`` (ranks começam em 1).

    Args:
        rankings: Mapeamento nome -> sequência de itens contendo ``doc_id``.
        k: Constante RRF positiva, usada para suavizar a contribuição de ranks baixos.
        weights: Pesos opcionais por ranking; por padrão todos recebem peso 1.
        top_k: Limite opcional de resultados retornados.

    Os scores originais dos métodos não participam do cálculo, pois têm escalas
    diferentes. Empates são resolvidos pelo ID do documento para manter resultado
    determinístico.
    """
    if k <= 0:
        raise ValueError('A constante k do RRF deve ser maior que zero.')
    if top_k is not None and top_k < 0:
        raise ValueError('top_k não pode ser negativo.')
    weights = weights or {}
    fused_scores = defaultdict(float)
    component_ranks = defaultdict(dict)

    for ranking_name, ranking in rankings.items():
        weight = float(weights.get(ranking_name, 1.0))
        if weight < 0:
            raise ValueError('Os pesos dos rankings devem ser não negativos.')
        seen = set()
        for rank, item in enumerate(ranking, start=1):
            doc_id = item['doc_id']
            if doc_id in seen:
                continue
            seen.add(doc_id)
            fused_scores[doc_id] += weight / (k + rank)
            component_ranks[doc_id][ranking_name] = rank

    ordered = sorted(fused_scores, key=lambda doc_id: (-fused_scores[doc_id], doc_id))
    if top_k is not None:
        ordered = ordered[:top_k]
    return [
        {'doc_id': doc_id, 'score': float(fused_scores[doc_id]),
         'component_ranks': component_ranks[doc_id]}
        for doc_id in ordered
    ]


class HybridRRFRetriever:
    """Combina um recuperador lexical e outro semântico por RRF."""
    def __init__(self, lexical_retriever, semantic_retriever,
                 k=RRF_K, candidate_depth=HYBRID_CANDIDATE_DEPTH,
                 lexical_weight=1.0, semantic_weight=1.0, document_ids=None):
        if candidate_depth <= 0:
            raise ValueError('candidate_depth deve ser maior que zero.')
        self.lexical_retriever = lexical_retriever
        self.semantic_retriever = semantic_retriever
        self.document_ids = list(document_ids if document_ids is not None else
                                  getattr(semantic_retriever, 'document_ids', []))
        if not self.document_ids:
            raise ValueError('Informe document_ids para completar o Top-100 lexical.')
        self.k = k
        self.candidate_depth = candidate_depth
        self.weights = {
            'lexical_bm25': lexical_weight,
            'semantic_all-MiniLM-L6-v2': semantic_weight,
        }

    def search(self, query, top_k=10):
        """Busca candidatos nos dois métodos e retorna o ranking híbrido."""
        if top_k <= 0:
            return []
        depth = max(top_k, self.candidate_depth)
        lexical = self.lexical_retriever.search(query, top_k=depth)
        semantic = self.semantic_retriever.search(query, top_k=depth)
        rankings = {
            'lexical_bm25': complete_ranking(lexical, self.document_ids, depth),
            'semantic_all-MiniLM-L6-v2': complete_ranking(semantic, self.document_ids, depth),
        }
        return reciprocal_rank_fusion(rankings, k=self.k, weights=self.weights, top_k=top_k)
