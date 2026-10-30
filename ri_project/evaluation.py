"""Métricas, rankings e comparações entre recuperadores."""
import json
import math
from pathlib import Path
from time import perf_counter

import pandas as pd

from .config import (EVALUATION_K, EVALUATION_RANKING_DEPTH, EVALUATION_SPLIT,
                     EXPERIMENTS_DIR)
from .preprocessing import preprocess
from .hybrid import complete_ranking, reciprocal_rank_fusion

BASELINE_METHODS = ('lexical_bm25', 'semantic_all-MiniLM-L6-v2')
HYBRID_METHOD = 'hybrid_rrf'
RERANK_METHOD = 'hybrid_reranking'
EVALUATION_METHODS = BASELINE_METHODS + (HYBRID_METHOD, RERANK_METHOD)


def evaluation_judgments(qrels_frame):
    judgments = {}
    for _, row in qrels_frame.iterrows():
        qid, doc_id, grade = row['query-id'], row['corpus-id'], float(row['score'])
        per_query = judgments.setdefault(qid, {})
        per_query[doc_id] = max(grade, per_query.get(doc_id, float('-inf')))
    return judgments


def retrieval_metrics(ranked_doc_ids, relevance_grades, k=10):
    """P/R/AP e NDCG com ganho linear e denominadores dos qrels completos."""
    if k <= 0:
        raise ValueError('k deve ser positivo.')
    if len(set(ranked_doc_ids[:k])) != len(ranked_doc_ids[:k]):
        raise ValueError('O ranking não pode repetir documentos na janela avaliada.')
    relevant_ids = {doc_id for doc_id, grade in relevance_grades.items() if grade > 0}
    retrieved = ranked_doc_ids[:k]
    relevant_hits = [doc_id in relevant_ids for doc_id in retrieved]
    precision = sum(relevant_hits) / k
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


def tune_rrf_k(queries, qrels, bm25, semantic, document_ids,
               output_dir=EXPERIMENTS_DIR, split='dev',
               k_values=(10, 60, 100), candidate_depth=100, cutoff=10):
    """Seleciona k pelo maior NDCG@10 médio em dev, fundindo Top-100 de cada método."""
    if split != 'dev':
        raise ValueError('O ajuste de k do RRF deve usar exclusivamente split=dev.')
    if not k_values or any(k <= 0 for k in k_values):
        raise ValueError('Informe valores positivos de k para o RRF.')
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    judgments = evaluation_judgments(qrels[split])
    query_ids = sorted(set(judgments) & set(queries))
    if not query_ids:
        raise ValueError('Não há consultas de dev em comum entre qrels e queries.jsonl.')

    component_rankings = {}
    for query_id in query_ids:
        text = str(queries[query_id].get('text', '') or '')
        lexical = bm25.search(text, top_k=candidate_depth)
        semantic_ranking = semantic.search(text, top_k=candidate_depth)
        component_rankings[query_id] = {
            'lexical_bm25': complete_ranking(lexical, document_ids, candidate_depth),
            'semantic_all-MiniLM-L6-v2': complete_ranking(
                semantic_ranking, document_ids, candidate_depth),
        }

    summary_rows, per_query_rows = [], []
    for k in k_values:
        scores = []
        for query_id in query_ids:
            fused = reciprocal_rank_fusion(component_rankings[query_id], k=k,
                                           top_k=candidate_depth)
            score = retrieval_metrics([item['doc_id'] for item in fused],
                                      judgments[query_id], cutoff)['NDCG10']
            scores.append(score)
            per_query_rows.append({'query_id': query_id, 'k': k, 'NDCG10': score})
        summary_rows.append({'k': k, 'split': split, 'query_count': len(query_ids),
                             'NDCG10': sum(scores) / len(scores),
                             'candidate_depth_per_method': candidate_depth})

    summary = pd.DataFrame(summary_rows).sort_values(
        ['NDCG10', 'k'], ascending=[False, True], kind='stable').reset_index(drop=True)
    per_query = pd.DataFrame(per_query_rows)
    summary.to_csv(output_dir / 'rrf_dev_tuning.csv', index=False)
    per_query.to_csv(output_dir / 'rrf_dev_tuning_per_query.csv', index=False)
    return summary, per_query, int(summary.iloc[0]['k'])


def run_evaluation(documents, queries, qrels, bm25, semantic, hybrid=None, reranked=None,
                   output_dir=EXPERIMENTS_DIR, split=EVALUATION_SPLIT,
                   cutoff=EVALUATION_K, ranking_depth=EVALUATION_RANKING_DEPTH):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    judgments = evaluation_judgments(qrels[split])
    query_ids = sorted(set(judgments) & set(queries))
    if not query_ids:
        raise ValueError('Não há consultas em comum entre qrels e queries.jsonl')
    ordered_doc_ids = sorted(documents)
    methods = BASELINE_METHODS
    if hybrid is not None:
        methods += (HYBRID_METHOD,)
    if reranked is not None:
        methods += (RERANK_METHOD,)
    ranking_records = {m: [] for m in methods}
    metric_records, query_time_totals, rankings = [], {m: 0.0 for m in methods}, {}
    retrieval_time_totals = {m: 0.0 for m in methods}
    rerank_time_totals = {m: 0.0 for m in methods}
    retrievers = {'lexical_bm25': bm25, 'semantic_all-MiniLM-L6-v2': semantic}
    if hybrid is not None:
        retrievers[HYBRID_METHOD] = hybrid
    if reranked is not None:
        retrievers[RERANK_METHOD] = reranked
    for qid in query_ids:
        query_text = str(queries[qid].get('text', '') or '')
        grades = judgments[qid]
        rankings[qid] = {}
        for method, retriever in retrievers.items():
            start = perf_counter()
            ranked = retriever.search(query_text, top_k=ranking_depth)
            if method == 'lexical_bm25':
                ranked = complete_ranking(ranked, ordered_doc_ids, ranking_depth)
            elapsed = perf_counter() - start
            query_time_totals[method] += elapsed
            if method == RERANK_METHOD:
                retrieval_time_totals[method] += reranked.last_hybrid_retrieval_seconds
                rerank_time_totals[method] += reranked.reranker.last_reranking_seconds
            else:
                retrieval_time_totals[method] += elapsed
            rankings[qid][method] = ranked
            metrics = retrieval_metrics([item['doc_id'] for item in ranked], grades, cutoff)
            metric_records.append({'query_id': qid, 'model': method, **metrics})
            for rank, item in enumerate(ranked, start=1):
                record = {'query_id': qid, 'doc_id': item['doc_id'],
                    'rank': rank, 'score': float(item['score']),
                    'relevance': float(grades.get(item['doc_id'], 0.0))}
                if method == HYBRID_METHOD:
                    component_ranks = item.get('component_ranks', {})
                    record['lexical_rank'] = component_ranks.get('lexical_bm25')
                    record['semantic_rank'] = component_ranks.get('semantic_all-MiniLM-L6-v2')
                if method == RERANK_METHOD:
                    record['hybrid_rank'] = item.get('hybrid_rank')
                    record['hybrid_score'] = item.get('hybrid_score')
                    record['reranker_score'] = item.get('reranker_score')
                ranking_records[method].append(record)
    per_query = pd.DataFrame(metric_records)
    per_query.to_csv(output_dir / 'metrics_per_query.csv', index=False)
    paths = {'lexical_bm25': output_dir / 'results_lexical.csv',
             'semantic_all-MiniLM-L6-v2': output_dir / 'results_semantic.csv'}
    if hybrid is not None:
        paths[HYBRID_METHOD] = output_dir / 'results_hybrid.csv'
    if reranked is not None:
        paths[RERANK_METHOD] = output_dir / 'results_reranking.csv'
    for method, records in ranking_records.items():
        columns = ['query_id', 'doc_id', 'rank', 'score', 'relevance']
        if method == HYBRID_METHOD:
            columns.extend(['lexical_rank', 'semantic_rank'])
        if method == RERANK_METHOD:
            columns.extend(['hybrid_rank', 'hybrid_score', 'reranker_score'])
        pd.DataFrame(records, columns=columns).to_csv(paths[method], index=False)
    summary = pd.DataFrame([{
        'split': split, 'model': method,
        'query_count': int((per_query['model'] == method).sum()),
        'P10': per_query.loc[per_query.model == method, 'P10'].mean(),
        'R10': per_query.loc[per_query.model == method, 'R10'].mean(),
        'MAP10': per_query.loc[per_query.model == method, 'AP10'].mean(),
        'NDCG10': per_query.loc[per_query.model == method, 'NDCG10'].mean(),
        'average_query_time_seconds': query_time_totals[method] / len(query_ids),
        'ranking_depth': ranking_depth, 'evaluation_cutoff': cutoff,
        'average_retrieval_time_seconds': retrieval_time_totals[method] / len(query_ids),
        'average_reranking_time_seconds': rerank_time_totals[method] / len(query_ids),
    } for method in methods])
    summary.to_csv(output_dir / 'metrics_summary.csv', index=False)
    if any(len(rankings[qid]['lexical_bm25']) != ranking_depth for qid in query_ids):
        raise AssertionError('O ranking lexical não contém a profundidade configurada para todas as consultas.')
    if any(len(rankings[qid]['semantic_all-MiniLM-L6-v2']) != ranking_depth for qid in query_ids):
        raise AssertionError('O ranking semântico não contém a profundidade configurada para todas as consultas.')
    if hybrid is not None and any(len(rankings[qid][HYBRID_METHOD]) != ranking_depth for qid in query_ids):
        raise AssertionError('O ranking híbrido não contém a profundidade configurada para todas as consultas.')
    if reranked is not None and any(len(rankings[qid][RERANK_METHOD]) != ranking_depth for qid in query_ids):
        raise AssertionError('O ranking reranqueado não contém a profundidade configurada para todas as consultas.')
    return {'metrics_per_query': per_query, 'metrics_summary': summary,
            'rankings_by_query': rankings, 'judgments': judgments,
            'query_ids': query_ids, 'ranking_paths': paths}


def build_comparison_cases(documents, queries, result, output_dir=EXPERIMENTS_DIR):
    """Exporta seis casos pareados e evidências lexicais para interpretação."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    per_query = result['metrics_per_query']
    judgments, rankings = result['judgments'], result['rankings_by_query']
    # Os casos explicativos seguem comparando os dois baselines pareados;
    # métricas híbridas aparecem na avaliação geral sem alterar essa seleção.
    baseline_rows = per_query[per_query['model'].isin(BASELINE_METHODS)]
    wide = baseline_rows.pivot(index='query_id', columns='model', values='NDCG10')
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
        for method in BASELINE_METHODS:
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


def build_reranking_analysis(documents, queries, result, output_dir=EXPERIMENTS_DIR):
    """Seleciona casos do item 10 e exporta movimentos de posição em test.

    Seleciona os dois maiores ganhos de NDCG@10, o caso mais próximo de zero
    (desempatando pelo menor número de documentos cuja mudança envolve o Top-10)
    e a maior queda. A comparação usa rankings pareados do resultado da
    avaliação, sem ajustar parâmetros ou modelos com base em test.
    """
    per_query = result['metrics_per_query']
    rankings = result['rankings_by_query']
    required = {HYBRID_METHOD, RERANK_METHOD}
    if not required.issubset(set(per_query['model'])):
        raise ValueError('A análise requer os resultados híbrido e reranqueado.')

    wide = per_query[per_query.model.isin(required)].pivot(
        index='query_id', columns='model', values='NDCG10')
    wide['delta_ndcg10'] = wide[RERANK_METHOD] - wide[HYBRID_METHOD]
    query_ids = [qid for qid in wide.index if qid in rankings and qid in queries]
    if len(query_ids) < 4:
        raise ValueError('A análise requer pelo menos quatro consultas pareadas.')
    wide = wide.loc[query_ids]

    positive = wide[wide.delta_ndcg10 > 0].sort_values(
        ['delta_ndcg10'], ascending=False, kind='stable')
    negative = wide[wide.delta_ndcg10 < 0].sort_values(
        ['delta_ndcg10'], ascending=True, kind='stable')
    if len(positive) < 2 or negative.empty:
        raise ValueError('Não há casos suficientes de ganho e piora para o item 10.')
    chosen = list(positive.index[:2])
    candidates = wide.drop(index=chosen + [negative.index[0]])
    if candidates.empty:
        raise ValueError('Não há consulta restante para o caso de pouca alteração.')

    def top10_changes(qid):
        before = {item['doc_id']: rank for rank, item in enumerate(
            rankings[qid][HYBRID_METHOD], start=1)}
        after = {item['doc_id']: rank for rank, item in enumerate(
            rankings[qid][RERANK_METHOD], start=1)}
        return sum((before.get(doc_id, 10**9) != after.get(doc_id, 10**9))
                   and (before.get(doc_id, 10**9) <= 10 or after.get(doc_id, 10**9) <= 10)
                   for doc_id in set(before) | set(after))

    candidates = candidates.assign(
        _abs_delta=candidates.delta_ndcg10.abs(),
        _top10_changes=[top10_changes(qid) for qid in candidates.index])
    unchanged_qid = candidates.sort_values(
        ['_abs_delta', '_top10_changes'], ascending=[True, True],
        kind='stable').index[0]
    case_ids = chosen + [unchanged_qid, negative.index[0]]
    labels = ['maior ganho', 'segundo maior ganho', 'pouca ou nenhuma alteração',
              'maior piora']

    case_rows, movement_rows, top10_rows = [], [], []
    for label, qid in zip(labels, case_ids):
        before = rankings[qid][HYBRID_METHOD]
        after = rankings[qid][RERANK_METHOD]
        before_rank = {item['doc_id']: rank for rank, item in enumerate(before, start=1)}
        after_rank = {item['doc_id']: rank for rank, item in enumerate(after, start=1)}
        before_items = {item['doc_id']: item for item in before}
        after_items = {item['doc_id']: item for item in after}
        docs_in_union = set(before_rank) | set(after_rank)
        grades = result['judgments'][qid]
        query_text = str(queries[qid].get('text', '') or '')
        case_rows.append({
            'case': label, 'query_id': qid, 'query': query_text,
            'ndcg10_hybrid': float(wide.loc[qid, HYBRID_METHOD]),
            'ndcg10_reranking': float(wide.loc[qid, RERANK_METHOD]),
            'delta_ndcg10': float(wide.loc[qid, 'delta_ndcg10']),
            'changed_documents_in_top10_union': top10_changes(qid),
        })
        for doc_id in docs_in_union:
            old_rank, new_rank = before_rank.get(doc_id), after_rank.get(doc_id)
            if old_rank == new_rank:
                continue
            # O arquivo contém movimentações que afetam a janela avaliada.
            if old_rank is None or new_rank is None or min(old_rank, new_rank) > 10:
                continue
            doc = documents[doc_id]
            title = str(doc.get('title', '') or '')
            text = str(doc.get('text', '') or '')
            movement_rows.append({
                'case': label, 'query_id': qid, 'query': query_text,
                'doc_id': doc_id, 'hybrid_rank': old_rank, 'reranked_rank': new_rank,
                'rank_change': new_rank - old_rank,
                'relevance': float(grades.get(doc_id, 0.0)),
                'hybrid_score': float(before_items[doc_id]['score']) if doc_id in before_items else None,
                'reranker_score': float(after_items[doc_id]['score']) if doc_id in after_items else None,
                'title': title, 'text': text,
            })
        for method, items, rank_map in (
            ('hybrid_rrf', before, before_rank), ('hybrid_reranking', after, after_rank)):
            for doc_id, rank in rank_map.items():
                if rank > 10:
                    continue
                item = (before_items if method == 'hybrid_rrf' else after_items)[doc_id]
                doc = documents[doc_id]
                top10_rows.append({
                    'case': label, 'query_id': qid, 'query': query_text,
                    'method': method, 'rank': rank, 'doc_id': doc_id,
                    'relevance': float(grades.get(doc_id, 0.0)),
                    'score': float(item['score']),
                    'title': str(doc.get('title', '') or ''),
                })

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cases = pd.DataFrame(case_rows)
    movements = pd.DataFrame(movement_rows).sort_values(
        ['case', 'reranked_rank', 'hybrid_rank'], kind='stable')
    top10 = pd.DataFrame(top10_rows).sort_values(
        ['case', 'method', 'rank'], kind='stable')
    cases.to_csv(output_dir / 'reranking_analysis_cases.csv', index=False)
    movements.to_csv(output_dir / 'reranking_analysis_movements.csv', index=False)
    top10.to_csv(output_dir / 'reranking_analysis_top10.csv', index=False)
    return {'cases': cases, 'movements': movements, 'top10': top10}
