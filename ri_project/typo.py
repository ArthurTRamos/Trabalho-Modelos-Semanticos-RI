"""Correção de erros com k-gramas, Jaccard e distância de edição."""
import json
import math
import re
import string
from collections import defaultdict

import pandas as pd

from .preprocessing import tokenize

KGRAM_SIZE = 3
JACCARD_THRESHOLD = 0.30
MAX_EDIT_DISTANCE = 2
TYPO_OPERATIONS = ('insertion', 'deletion', 'substitution', 'transposition')
TYPO_LABELS = {
    'insertion': 'inserção', 'deletion': 'remoção',
    'substitution': 'substituição', 'transposition': 'troca de ordem',
}


def character_kgrams(term, k=KGRAM_SIZE):
    if k <= 0:
        raise ValueError('k deve ser maior que zero')
    padded = '$' * (k - 1) + term + '$' * (k - 1)
    return {padded[i:i + k] for i in range(max(1, len(padded) - k + 1))}


def build_kgram_index(vocabulary, k=KGRAM_SIZE):
    index = defaultdict(set)
    for term in vocabulary:
        for gram in character_kgrams(term, k):
            index[gram].add(term)
    return dict(index)


def jaccard_coefficient(left, right):
    union = left | right
    return len(left & right) / len(union) if union else 1.0


def edit_distance(left, right):
    """Distância de Levenshtein com custo unitário para cada edição."""
    if len(left) < len(right):
        left, right = right, left
    previous = list(range(len(right) + 1))
    for i, left_char in enumerate(left, start=1):
        current = [i]
        for j, right_char in enumerate(right, start=1):
            current.append(min(current[-1] + 1, previous[j] + 1,
                               previous[j - 1] + (left_char != right_char)))
        previous = current
    return previous[-1]


def best_spelling_candidate(term, vocabulary, candidate_index,
                            jaccard_threshold=JACCARD_THRESHOLD,
                            max_distance=MAX_EDIT_DISTANCE, k=KGRAM_SIZE):
    if term in vocabulary:
        return None
    query_grams = character_kgrams(term, k)
    candidates = set().union(*(candidate_index.get(g, set()) for g in query_grams))
    filtered = []
    for candidate in candidates:
        similarity = jaccard_coefficient(query_grams, character_kgrams(candidate, k))
        if similarity >= jaccard_threshold:
            distance = edit_distance(term, candidate)
            if distance <= max_distance:
                filtered.append((distance, -similarity, candidate, similarity))
    if not filtered:
        return None
    distance, _, candidate, similarity = min(filtered)
    return {'term': candidate, 'distance': distance, 'jaccard': similarity}


def correct_query(query, vocabulary, candidate_index,
                  jaccard_threshold=JACCARD_THRESHOLD,
                  max_distance=MAX_EDIT_DISTANCE, k=KGRAM_SIZE):
    corrections = []
    def replace_unknown(match):
        observed = match.group(0).lower()
        choice = best_spelling_candidate(observed, vocabulary, candidate_index,
                                         jaccard_threshold, max_distance, k)
        if choice is None:
            return match.group(0)
        corrections.append({'observed': observed, 'replacement': choice['term'],
                            'edit_distance': choice['distance'], 'jaccard': choice['jaccard']})
        return choice['term']
    return re.sub(r'\b\w+\b', replace_unknown, str(query)), corrections


def _typo_candidates(term, operation):
    alphabet = string.ascii_lowercase
    if operation == 'insertion':
        return [term[:i] + c + term[i:] for i in range(len(term) + 1) for c in alphabet]
    if operation == 'deletion':
        return [term[:i] + term[i + 1:] for i in range(len(term)) if len(term) > 2]
    if operation == 'substitution':
        return [term[:i] + c + term[i + 1:] for i in range(len(term))
                for c in alphabet if c != term[i]]
    if operation == 'transposition':
        return [term[:i] + term[i + 1] + term[i] + term[i + 2:]
                for i in range(len(term) - 1) if term[i] != term[i + 1]]
    raise ValueError(f'Tipo de erro desconhecido: {operation}')


def make_oov_typo(query, operation, vocabulary):
    for original in sorted(set(tokenize(query)), key=lambda term: (-len(term), term)):
        if original not in vocabulary:
            continue
        for typo in _typo_candidates(original, operation):
            if typo not in vocabulary:
                pattern = rf'\b{re.escape(original)}\b'
                typo_query = re.sub(pattern, typo, query, count=1, flags=re.IGNORECASE)
                if typo_query != query:
                    return {'query': typo_query, 'original_term': original, 'typo_term': typo}
    return None


def _qrels_by_query(qrels_frame):
    judgments = defaultdict(dict)
    for _, row in qrels_frame.iterrows():
        qid, doc_id = row['query-id'], row['corpus-id']
        judgments[qid][doc_id] = max(float(row['score']), judgments[qid].get(doc_id, float('-inf')))
    return dict(judgments)


def ndcg_at_k(ranked_doc_ids, relevant_grades, k=10):
    if k <= 0:
        return 0.0
    dcg = sum(relevant_grades.get(doc_id, 0.0) / math.log2(rank + 2)
              for rank, doc_id in enumerate(ranked_doc_ids[:k]))
    ideal = sorted(relevant_grades.values(), reverse=True)[:k]
    idcg = sum(grade / math.log2(rank + 2) for rank, grade in enumerate(ideal))
    return dcg / idcg if idcg else 0.0


class TypoExperiment:
    """Coordena ajuste em dev e avaliação em test sem dependência de variáveis globais."""
    def __init__(self, vocabulary, queries, qrels, retriever):
        self.vocabulary, self.queries, self.qrels, self.retriever = vocabulary, queries, qrels, retriever

    def _ndcg(self, text, qid, judgments, k):
        ranked = self.retriever.search(text, top_k=k)
        return ndcg_at_k([item['doc_id'] for item in ranked], judgments.get(qid, {}), k)

    def prepare_cases(self, n_queries=10, split='dev', k=10):
        if split not in self.qrels:
            raise ValueError(f'Partição desconhecida: {split}. Use train, dev ou test.')
        judgments = _qrels_by_query(self.qrels[split])
        qids = sorted(set(judgments) & set(self.queries))
        cases = []
        for qid in qids:
            if len(cases) == n_queries:
                break
            original = self.queries[qid].get('text', '')
            original_ndcg = self._ndcg(original, qid, judgments, k)
            if original_ndcg <= 0:
                continue
            operation = TYPO_OPERATIONS[len(cases) % len(TYPO_OPERATIONS)]
            typo = make_oov_typo(original, operation, self.vocabulary)
            if typo is None:
                continue
            error_ndcg = self._ndcg(typo['query'], qid, judgments, k)
            cases.append({'query_id': qid, 'query_original': original,
                          'query_erro': typo['query'], 'tipo_erro': operation,
                          'token_original': typo['original_term'], 'token_com_erro': typo['typo_term'],
                          'NDCG10_original': original_ndcg, 'NDCG10_erro': error_ndcg})
        if len(cases) != n_queries:
            raise RuntimeError(f'Foram encontradas apenas {len(cases)} consultas elegíveis em {split}; eram necessárias {n_queries}.')
        return cases

    def tune(self, n_queries=10, k=10, kgram_sizes=(2, 3, 4),
             jaccard_thresholds=(.20, .30, .40, .50), max_edit_distances=(1, 2)):
        cases = self.prepare_cases(n_queries, 'dev', k)
        judgments = _qrels_by_query(self.qrels['dev'])
        rows = []
        for gram_size in kgram_sizes:
            candidate_index = build_kgram_index(self.vocabulary, gram_size)
            for threshold in jaccard_thresholds:
                for max_distance in max_edit_distances:
                    recoveries, target_hits, corrected_scores, damaged = [], 0, [], 0
                    for case in cases:
                        corrected, corrections = correct_query(case['query_erro'], self.vocabulary,
                            candidate_index, threshold, max_distance, gram_size)
                        score = self._ndcg(corrected, case['query_id'], judgments, k)
                        corrected_scores.append(score)
                        target_hits += any(c['observed'] == case['token_com_erro'] and
                                           c['replacement'] == case['token_original'] for c in corrections)
                        loss = case['NDCG10_original'] - case['NDCG10_erro']
                        if loss > 0:
                            damaged += 1
                            recoveries.append((score - case['NDCG10_erro']) / loss)
                    rows.append({'kgram_size': gram_size, 'jaccard_threshold': threshold,
                        'max_edit_distance': max_distance,
                        'mean_recovery_fraction': sum(recoveries) / len(recoveries) if recoveries else 0.,
                        'target_correction_accuracy': target_hits / len(cases),
                        'mean_corrected_ndcg10': sum(corrected_scores) / len(corrected_scores),
                        'num_dev_cases': len(cases), 'num_cases_with_performance_loss': damaged})
        rows.sort(key=lambda r: (-r['target_correction_accuracy'], -r['mean_corrected_ndcg10'],
            -r['mean_recovery_fraction'], abs(r['kgram_size'] - 3), -r['jaccard_threshold'], r['max_edit_distance']))
        return pd.DataFrame(rows), rows[0]

    def run(self, n_queries=10, split='test', k=10, jaccard_threshold=JACCARD_THRESHOLD,
            max_distance=MAX_EDIT_DISTANCE, kgram_size=KGRAM_SIZE):
        if split == 'test' and n_queries != 10:
            raise ValueError('O protocolo do PDF exige exatamente 10 consultas de test.')
        if n_queries <= 0:
            raise ValueError('n_queries deve ser maior que zero')
        judgments = _qrels_by_query(self.qrels[split])
        cases = self.prepare_cases(n_queries, split, k)
        candidate_index = build_kgram_index(self.vocabulary, kgram_size)
        records = []
        for case in cases:
            corrected, corrections = correct_query(case['query_erro'], self.vocabulary,
                candidate_index, jaccard_threshold, max_distance, kgram_size)
            score = self._ndcg(corrected, case['query_id'], judgments, k)
            denominator = case['NDCG10_original'] - case['NDCG10_erro']
            target = [c for c in corrections if c['observed'] == case['token_com_erro']]
            records.append({'query_id': case['query_id'], 'query_original': case['query_original'],
                'query_erro': case['query_erro'], 'query_corrigida': corrected,
                'tipo_erro': TYPO_LABELS[case['tipo_erro']], 'token_original': case['token_original'],
                'token_com_erro': case['token_com_erro'],
                'token_corrigido': target[0]['replacement'] if target else '',
                'correcoes_aplicadas': json.dumps(corrections, ensure_ascii=False),
                'NDCG10_original': case['NDCG10_original'], 'NDCG10_erro': case['NDCG10_erro'],
                'NDCG10_corrigida': score,
                'fracao_desempenho_recuperada': (score - case['NDCG10_erro']) / denominator if denominator > 0 else None,
                'status_recuperacao': 'calculada' if denominator > 0 else 'erro não reduziu o desempenho'})
        return pd.DataFrame(records)
