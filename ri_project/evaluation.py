"""Métricas, rankings e comparações entre recuperadores."""
import json
import math
from time import perf_counter

import pandas as pd

from .config import (EVALUATION_K, EVALUATION_RANKING_DEPTH, EVALUATION_SPLIT,
                     EXPERIMENTS_DIR)
from .preprocessing import preprocess

EVALUATION_METHODS = ('lexical_bm25', 'semantic_all-MiniLM-L6-v2')


def evaluation_judgments(qrels_frame):
    judgments = {}
    for _, row in qrels_frame.iterrows():
        qid, doc_id, grade = row['query-id'], row['corpus-id'], float(row['score'])
        per_query = judgments.setdefault(qid, {})
        per_query[doc_id] = max(grade, per_query.get(doc_id, float('-inf')))
    return judgments


def retrieval_metrics(ranked_doc_ids, relevance_grades, k=10):
    relevant_ids = {doc_id for doc_id, grade in relevance_grades.items() if grade > 0}
    retrieved = ranked_doc_ids[:k]
    relevant_hits = [doc_id in relevant_ids for doc_id in retrieved]
    precision = sum(relevant_hits) / k if k else 0.0
    recall = sum(relevant_hits) / len(relevant_ids) if relevant_ids else 0.0
    precision_sum = seen = 0.0
    for rank, doc_id in enumerate(retrieved, start=1):
        if doc_id in relevant_ids:
            seen += 1
            precision_sum += seen / rank
    ap = precision_sum / len(relevant_ids) if relevant_ids else 0.0
    dcg = sum(relevance_grades.get(doc_id, 0.0) / math.log2(rank + 1)
              for rank, doc_id in enumerate(retrieved, start=1))
    ideal = sorted(relevance_grades.values(), reverse=True)[:k]
    idcg = sum(grade / math.log2(rank + 1) for rank, grade in enumerate(ideal, start=1))
    return {'P10': precision, 'R10': recall, 'AP10': ap, 'NDCG10': dcg / idcg if idcg else 0.0}


def run_evaluation(documents, queries, qrels, bm25, semantic,
                   output_dir=EXPERIMENTS_DIR, split=EVALUATION_SPLIT,
                   cutoff=EVALUATION_K, ranking_depth=EVALUATION_RANKING_DEPTH):
    judgments = evaluation_judgments(qrels[split])
    query_ids = sorted(set(judgments) & set(queries))
    if not query_ids:
        raise ValueError('Não há consultas em comum entre qrels e queries.jsonl')
    ordered_doc_ids = sorted(documents)
    ranking_records = {m: [] for m in EVALUATION_METHODS}
    metric_records, query_time_totals, rankings = [], {m: 0.0 for m in EVALUATION_METHODS}, {}
    retrievers = {'lexical_bm25': bm25, 'semantic_all-MiniLM-L6-v2': semantic}
    for qid in query_ids:
        query_text = str(queries[qid].get('text', '') or '')
        grades = judgments[qid]
        rankings[qid] = {}
        for method, retriever in retrievers.items():
            start = perf_counter()
            ranked = retriever.search(query_text, top_k=ranking_depth)
            if method == 'lexical_bm25' and len(ranked) < ranking_depth:
                present = {item['doc_id'] for item in ranked}
                for doc_id in ordered_doc_ids:
                    if doc_id not in present:
                        ranked.append({'doc_id': doc_id, 'score': 0.0})
                        if len(ranked) == ranking_depth:
                            break
            query_time_totals[method] += perf_counter() - start
            rankings[qid][method] = ranked
            metrics = retrieval_metrics([item['doc_id'] for item in ranked], grades, cutoff)
            metric_records.append({'query_id': qid, 'model': method, **metrics})
            for rank, item in enumerate(ranked, start=1):
                ranking_records[method].append({'query_id': qid, 'doc_id': item['doc_id'],
                    'rank': rank, 'score': float(item['score']),
                    'relevance': float(grades.get(item['doc_id'], 0.0))})
    per_query = pd.DataFrame(metric_records)
    per_query.to_csv(output_dir / 'metrics_per_query.csv', index=False)
    paths = {'lexical_bm25': output_dir / 'results_lexical.csv',
             'semantic_all-MiniLM-L6-v2': output_dir / 'results_semantic.csv'}
    for method, records in ranking_records.items():
        pd.DataFrame(records, columns=['query_id', 'doc_id', 'rank', 'score', 'relevance']).to_csv(paths[method], index=False)
    summary = pd.DataFrame([{
        'split': split, 'model': method,
        'query_count': int((per_query['model'] == method).sum()),
        'P10': per_query.loc[per_query.model == method, 'P10'].mean(),
        'R10': per_query.loc[per_query.model == method, 'R10'].mean(),
        'MAP10': per_query.loc[per_query.model == method, 'AP10'].mean(),
        'NDCG10': per_query.loc[per_query.model == method, 'NDCG10'].mean(),
        'average_query_time_seconds': query_time_totals[method] / len(query_ids),
        'ranking_depth': ranking_depth, 'evaluation_cutoff': cutoff,
    } for method in EVALUATION_METHODS])
    summary.to_csv(output_dir / 'metrics_summary.csv', index=False)
    if any(len(rankings[qid]['lexical_bm25']) != ranking_depth for qid in query_ids):
        raise AssertionError('O ranking lexical não contém a profundidade configurada para todas as consultas.')
    if any(len(rankings[qid]['semantic_all-MiniLM-L6-v2']) != ranking_depth for qid in query_ids):
        raise AssertionError('O ranking semântico não contém a profundidade configurada para todas as consultas.')
    return {'metrics_per_query': per_query, 'metrics_summary': summary,
            'rankings_by_query': rankings, 'judgments': judgments,
            'query_ids': query_ids, 'ranking_paths': paths}


def build_comparison_cases(documents, queries, result, output_dir=EXPERIMENTS_DIR):
    per_query = result['metrics_per_query']
    judgments, rankings = result['judgments'], result['rankings_by_query']
    wide = per_query.pivot(index='query_id', columns='model', values='NDCG10')
    wide['delta_ndcg10_semantic_minus_lexical'] = wide['semantic_all-MiniLM-L6-v2'] - wide['lexical_bm25']
    semantic_wins = wide[wide.delta_ndcg10_semantic_minus_lexical > 0].sort_values(
        'delta_ndcg10_semantic_minus_lexical', ascending=False).head(3)
    lexical_wins = wide[wide.delta_ndcg10_semantic_minus_lexical < 0].sort_values(
        'delta_ndcg10_semantic_minus_lexical').head(3)
    if len(semantic_wins) < 3 or len(lexical_wins) < 3:
        raise ValueError('O protocolo requer três consultas com vantagem estrita para cada método.')
    cases, top5_rows = [], []
    for qid in list(semantic_wins.index) + list(lexical_wins.index):
        query = str(queries[qid].get('text', '') or '')
        grades = judgments[qid]
        delta = float(wide.loc[qid, 'delta_ndcg10_semantic_minus_lexical'])
        winner = 'semantic_all-MiniLM-L6-v2' if delta > 0 else 'lexical_bm25'
        selected, evidence = {}, {}
        for method in EVALUATION_METHODS:
            items = rankings[qid][method][:5]
            selected[method] = [item['doc_id'] for item in items]
            evidence[method] = {}
            for rank, item in enumerate(items, start=1):
                doc_id = item['doc_id']
                document = documents[doc_id]
                title, text = str(document.get('title', '') or ''), str(document.get('text', '') or '')
                matches = sorted(set(preprocess(query)) & set(preprocess(f'{title} {text}')))
                evidence[method][doc_id] = matches
                top5_rows.append({'query_id': qid, 'query': query, 'winner_by_ndcg10': winner,
                    'delta_ndcg10_semantic_minus_lexical': delta,
                    'ndcg10_lexical': float(wide.loc[qid, 'lexical_bm25']),
                    'ndcg10_semantic': float(wide.loc[qid, 'semantic_all-MiniLM-L6-v2']),
                    'method': method, 'rank': rank, 'doc_id': doc_id,
                    'score': float(item['score']), 'relevance': float(grades.get(doc_id, 0.0)),
                    'is_relevant': grades.get(doc_id, 0.0) > 0,
                    'matched_bm25_query_terms': '|'.join(matches),
                    'matched_bm25_query_term_count': len(matches), 'title': title, 'text': text})
        cases.append({'query_id': qid, 'query': query, 'winner_by_ndcg10': winner,
            'delta_ndcg10_semantic_minus_lexical': delta,
            'ndcg10_lexical': float(wide.loc[qid, 'lexical_bm25']),
            'ndcg10_semantic': float(wide.loc[qid, 'semantic_all-MiniLM-L6-v2']),
            'lexical_top5_doc_ids': json.dumps(selected['lexical_bm25'], ensure_ascii=False),
            'lexical_relevant_top5_doc_ids': json.dumps([d for d in selected['lexical_bm25'] if grades.get(d, 0.) > 0], ensure_ascii=False),
            'semantic_top5_doc_ids': json.dumps(selected['semantic_all-MiniLM-L6-v2'], ensure_ascii=False),
            'semantic_relevant_top5_doc_ids': json.dumps([d for d in selected['semantic_all-MiniLM-L6-v2'] if grades.get(d, 0.) > 0], ensure_ascii=False),
            'lexical_top5_query_term_matches': json.dumps(evidence['lexical_bm25'], ensure_ascii=False),
            'semantic_top5_query_term_matches': json.dumps(evidence['semantic_all-MiniLM-L6-v2'], ensure_ascii=False)})
    cases_df, top5_df = pd.DataFrame(cases), pd.DataFrame(top5_rows)
    cases_df.to_csv(output_dir / 'comparison_cases_summary.csv', index=False)
    top5_df.to_csv(output_dir / 'comparison_cases_top5.csv', index=False)
    return cases_df, top5_df
