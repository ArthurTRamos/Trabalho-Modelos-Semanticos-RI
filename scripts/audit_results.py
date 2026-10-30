"""Audita CSVs de recuperação por fórmulas independentes, sem inferência neural."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from collections import defaultdict


def read_rows(path, delimiter=','):
    with Path(path).open(encoding='utf-8', newline='') as file:
        return list(csv.DictReader(file, delimiter=delimiter))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def close(left, right, message):
    require(math.isclose(float(left), float(right), abs_tol=1e-10, rel_tol=1e-9), message)


def audit(results, data):
    results, data = Path(results), Path(data)
    qrels = defaultdict(dict)
    for row in read_rows(data / 'qrels/test.tsv', '\t'):
        qrels[row['query-id']][row['corpus-id']] = float(row['score'])
    documents = {row['_id'] for row in map(json.loads, (data / 'corpus.jsonl').read_text().splitlines())}
    methods = {'lexical_bm25': 'results_lexical.csv',
               'semantic_all-MiniLM-L6-v2': 'results_semantic.csv',
               'hybrid_rrf': 'results_hybrid.csv', 'hybrid_reranking': 'results_reranking.csv'}
    metrics = read_rows(results / 'metrics_per_query.csv')
    require(len(metrics) == len(qrels) * 4, 'Quantidade de métricas incorreta.')
    measured = {(row['query_id'], row['model']): row for row in metrics}
    require(len(measured) == len(metrics), 'Métricas duplicadas.')
    rankings = {}
    for model, filename in methods.items():
        groups = defaultdict(list)
        for row in read_rows(results / filename):
            groups[row['query_id']].append(row)
        require(set(groups) == set(qrels), f'{filename}: consultas diferentes de test.')
        for qid, rows in groups.items():
            rows.sort(key=lambda row: int(row['rank']))
            ids = [row['doc_id'] for row in rows]
            require([int(row['rank']) for row in rows] == list(range(1, 101)), f'{qid}: ranks inválidos.')
            require(len(set(ids)) == 100 and set(ids) <= documents, f'{qid}: documentos inválidos.')
            for row in rows:
                close(row['relevance'], qrels[qid].get(row['doc_id'], 0), f'{qid}: grau divergente.')
                require(math.isfinite(float(row['score'])), f'{qid}: score não finito.')
            grades = qrels[qid]
            total = sum(grade > 0 for grade in grades.values())
            hits = [i for i, doc in enumerate(ids[:10], 1) if grades.get(doc, 0) > 0]
            ideal = sum(g / math.log2(i + 1) for i, g in enumerate(sorted(grades.values(), reverse=True)[:10], 1))
            dcg = sum(grades.get(doc, 0) / math.log2(i + 1) for i, doc in enumerate(ids[:10], 1))
            expected = {'P10': len(hits) / 10, 'R10': len(hits) / total if total else 0,
                        'AP10': sum(n / rank for n, rank in enumerate(hits, 1)) / total if total else 0,
                        'NDCG10': dcg / ideal if ideal else 0}
            for metric, value in expected.items():
                close(measured[qid, model][metric], value, f'{qid}/{model}/{metric}: divergência.')
        rankings[model] = groups
    for summary in read_rows(results / 'metrics_summary.csv'):
        model = summary['model']
        for label, source in [('P10', 'P10'), ('R10', 'R10'), ('MAP10', 'AP10'), ('NDCG10', 'NDCG10')]:
            average = sum(float(measured[qid, model][source]) for qid in qrels) / len(qrels)
            close(summary[label], average, f'{model}/{label}: média divergente.')
    tuning = read_rows(results / 'rrf_dev_tuning.csv')
    require({int(row['k']) for row in tuning} >= {10, 60, 100}, 'Faltam configurações RRF.')
    require(all(row['split'] == 'dev' for row in tuning), 'Tuning fora de dev.')
    selected = min(tuning, key=lambda row: (-float(row['NDCG10']), int(row['k'])))
    k = int(selected['k'])
    for qid in qrels:
        fused = defaultdict(float)
        for model in ('lexical_bm25', 'semantic_all-MiniLM-L6-v2'):
            for rank, row in enumerate(rankings[model][qid], 1):
                fused[row['doc_id']] += 1 / (k + rank)
        expected_ids = sorted(fused, key=lambda doc: (-fused[doc], doc))[:100]
        hybrid = rankings['hybrid_rrf'][qid]
        require([r['doc_id'] for r in hybrid] == expected_ids, f'{qid}: fusão divergente.')
        for row in hybrid:
            close(row['score'], fused[row['doc_id']], f'{qid}: score RRF divergente.')
        reranked = rankings['hybrid_reranking'][qid]
        after = [r['doc_id'] for r in reranked]
        require(set(after[:50]) == set(expected_ids[:50]), f'{qid}: candidatos de reranking divergentes.')
        require(after[50:] == expected_ids[50:], f'{qid}: cauda alterada.')
        scores = [float(row['reranker_score']) for row in reranked[:50]]
        require(scores == sorted(scores, reverse=True), f'{qid}: cross-encoder fora de ordem.')
    typos = read_rows(results / 'typo_experiment.csv')
    require(len(typos) == len({row['query_id'] for row in typos}) == 10, 'Exija dez consultas com erro.')
    require({row['tipo_erro'] for row in typos} == {'inserção', 'remoção', 'substituição', 'troca de ordem'}, 'Tipos de erro ausentes.')
    for row in typos:
        require(row['query_id'] in qrels and float(row['NDCG10_original']) > 0, 'Consulta de erro inelegível.')
        loss = float(row['NDCG10_original']) - float(row['NDCG10_erro'])
        if loss > 0:
            close(row['fracao_desempenho_recuperada'],
                  (float(row['NDCG10_corrigida']) - float(row['NDCG10_erro'])) / loss, 'Recuperação de erro incorreta.')
        else:
            require(row['status_recuperacao'] == 'erro não reduziu o desempenho', 'Estado de erro incorreto.')
    files = list(methods.values()) + ['metrics_per_query.csv', 'metrics_summary.csv',
                                     'rrf_dev_tuning.csv', 'typo_experiment.csv']
    return {'verified': True, 'test_queries_per_method': len(qrels), 'methods': len(methods),
            'rankings_per_method': len(qrels) * 100, 'selected_rrf_k': k,
            'metrics_recomputed_independently': True, 'reranking_tail_preserved': True,
            'typo_cases': len(typos), 'human_ratings': 'outside_scope',
            'input_sha256': {name: hashlib.sha256((results / name).read_bytes()).hexdigest() for name in files}}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, default=Path('Experiments'))
    parser.add_argument('--data', type=Path, default=Path('nfcorpus/nfcorpus'))
    args = parser.parse_args()
    report = audit(args.results, args.data)
    output = args.results / 'review' / 'retrieval_audit.json'
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(f'Auditoria concluída: {output}')
