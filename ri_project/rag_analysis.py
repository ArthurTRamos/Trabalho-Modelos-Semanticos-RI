"""Item 14: evidências por etapa e relatório final após revisão humana."""
import hashlib
import json
from pathlib import Path
import statistics

import pandas as pd

from .rag_evaluation import (COMMON_CRITERIA, RANKING_FILES, RAG_QUERY_IDS,
                             context_metrics, load_evaluation_artifacts, write_json)

CATEGORIES = ('melhora', 'pouca_diferenca', 'nao_ajudou')
STAGES = ('lexical', 'semantic', 'fusion', 'reranking', 'context_selection', 'generation', 'multiple', 'none')


def build_stage_diagnostics(artifacts, rankings_dir, output_dir):
    """Inspeciona rankings salvos; nunca usa qrels para alterar o contexto."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    by_stage = {}
    for stage, filename in RANKING_FILES.items():
        frame = pd.read_csv(Path(rankings_dir) / filename)
        by_stage[stage] = {}
        for qid in RAG_QUERY_IDS:
            group = frame[frame.query_id == qid].sort_values('rank')
            if group['rank'].tolist() != list(range(1, 101)) or group.doc_id.nunique() != 100:
                raise ValueError(f'{filename}/{qid}: o diagnóstico exige Top-100 único.')
            by_stage[stage][qid] = group.doc_id.tolist()
    summaries, doc_rows = [], []
    for case in artifacts['bundle']['cases']:
        qid = case['query_id']
        k = artifacts['bundle'].get('retrieval', {}).get('rrf_k')
        if k is None or k <= 0:
            raise ValueError('O diagnóstico exige o k RRF registrado no experimento.')
        # Confere a proveniência das posições dos baselines antes de usá-las
        # para atribuir falhas a uma etapa da recuperação.
        scores = {}
        weights = artifacts['bundle']['retrieval'].get('rrf_weights', {})
        for stage, name in (('lexical', 'lexical_bm25'), ('semantic', 'semantic_all-MiniLM-L6-v2')):
            for rank, doc_id in enumerate(by_stage[stage][qid], 1):
                scores[doc_id] = scores.get(doc_id, 0.0) + weights.get(name, 1.0) / (k + rank)
        fused = sorted(scores, key=lambda d: (-scores[d], d))[:100]
        if fused != by_stage['hybrid'][qid]:
            raise ValueError(f'RRF e baselines divergentes: {qid}; não misture execuções.')
        expected = [d['doc_id'] for d in case['reranked_top100']]
        if by_stage['reranking'][qid] != expected:
            raise ValueError(f'Ranking salvo e contexto real divergem em {qid}; não misture execuções.')
        ranks = {stage: {d: i for i, d in enumerate(by_stage[stage][qid], 1)} for stage in by_stage}
        final_ids = [d['doc_id'] for d in case['contexts']]
        grades = artifacts['judgments'][qid]
        relevant = {d for d, grade in grades.items() if grade > 0}
        hybrid = by_stage['hybrid'][qid]
        reranked = by_stage['reranking'][qid]
        if set(hybrid[:50]) != set(reranked[:50]) or hybrid[50:] != reranked[50:]:
            raise ValueError(f'Candidatos/cauda do reranking divergentes: {qid}.')
        row = {'query_id': qid, 'query': case['query'], 'total_relevant': len(relevant),
               'relevant_in_candidate_union': len(relevant & (set(by_stage['lexical'][qid]) | set(by_stage['semantic'][qid]))),
               'relevant_in_hybrid_top50': len(relevant & set(hybrid[:50])),
               'relevant_in_hybrid_top5': len(relevant & set(hybrid[:5])),
               'relevant_in_context': len(relevant & set(final_ids)),
               'context_is_actual_reranked_top5': True}
        for stage in by_stage:
            ids = by_stage[stage][qid]
            row[f'{stage}_relevant_top100'] = len(relevant & set(ids))
            row.update({f'{stage}_{m}': v for m, v in context_metrics(ids[:5], grades).items()
                        if m in ('P5', 'R5', 'Hit5')})
        summaries.append(row)
        doc_ids = relevant | set().union(*(set(ranks[stage]) for stage in ranks))
        for doc_id in sorted(doc_ids):
            doc = artifacts['documents'][doc_id]
            doc_rows.append({'query_id': qid, 'doc_id': doc_id,
                             'relevance': grades.get(doc_id, 0.0), 'has_qrel': doc_id in grades,
                             **{stage + '_rank': ranks[stage].get(doc_id) for stage in ranks},
                             'in_context': doc_id in final_ids,
                             'title': doc['title'], 'text': doc['text']})
    summary, documents = pd.DataFrame(summaries), pd.DataFrame(doc_rows)
    summary.to_csv(output_dir / 'stage_metrics_per_query.csv', index=False)
    documents.to_csv(output_dir / 'stage_document_evidence.csv', index=False)
    return {'summary': summary, 'documents': documents}


def prepare_case_review_template(artifacts, output_dir, selection=None):
    path = Path(output_dir) / 'case_reviews.json'
    if path.exists():
        review = json.loads(path.read_text())
        if review['artifact_fingerprint'] != artifacts['fingerprint']:
            raise ValueError('Revisão dos casos pertence a outro experimento.')
    else:
        review = {'artifact_fingerprint': artifacts['fingerprint'], 'reviewer_id': '', 'cases': [
            {'category': category, 'query_id': '', 'selection_reason': '', 'comparison': '',
             'information_in_context': 'pending', 'primary_stage': '', 'stage_reason': '',
             'evidence': [{'doc_id': '', 'quote': ''}]}
            for category in CATEGORIES]}
    if selection:
        for row in review['cases']:
            suggested = next((c for c in selection['cases'] if c['category'] == row['category']), None)
            if not row['query_id'] and suggested:
                row['query_id'] = suggested['query_id']
    write_json(path, review)
    return path


def write_analysis_dossiers(artifacts, diagnostics, output_dir):
    """Dossiers de todas as consultas, sem supor melhora a partir de relevância."""
    output_dir = Path(output_dir)
    parts = ['# Dossiers para análise de falhas — item 14\n',
             'Evidências automáticas; classificação qualitativa e avaliação humana pendentes. '
             'Relevância dos qrels não prova suporte a cada afirmação. As posições provêm '
             'dos rankings conferidos contra o contexto efetivamente enviado.\n']
    for case in artifacts['bundle']['cases']:
        qid = case['query_id']
        s = diagnostics['summary'].set_index('query_id').loc[qid]
        parts.extend([f'## {qid} — {case["query"]}\n',
                      f'Relevantes no universo lexical/semântico Top-100: {s.relevant_in_candidate_union}; '
                      f'no Top-50 híbrido: {s.relevant_in_hybrid_top50}; '
                      f'no Top-5 antes/depois do reranking: {s.relevant_in_hybrid_top5}/{s.relevant_in_context}.\n',
                      '| Documento enviado | Grau | Lexical | Semântico | RRF | Reranking |',
                      '|---|---:|---:|---:|---:|---:|'])
        for d in case['contexts']:
            row = diagnostics['documents'][(diagnostics['documents'].query_id == qid)
                                            & (diagnostics['documents'].doc_id == d['doc_id'])].iloc[0]
            positions = ['—' if pd.isna(row[f'{stage}_rank']) else str(int(row[f'{stage}_rank']))
                         for stage in RANKING_FILES]
            parts.append('| ' + d['doc_id'] + f' | {row.relevance:g} | ' + ' | '.join(positions) + ' |')
        for condition in ('A', 'B'):
            row = artifacts['pairs'][qid, condition]
            description = 'Nenhum contexto externo foi fornecido à LLM.' if condition == 'A' else 'Top-5 abaixo foi fornecido integralmente.'
            parts.extend([f'\n### Resposta {condition}\n\n{description}\n\n{row["answer"]}\n'])
        parts.append('### Contexto efetivamente enviado em B e referência para avaliar A\n')
        for d in case['contexts']:
            parts.append(f'#### {d["doc_id"]} — {d["title"]}\n\n{d["text"]}\n')
    (output_dir / 'failure_analysis_dossiers.md').write_text('\n'.join(parts), encoding='utf-8')
    prepare_case_review_template(artifacts, output_dir)
    status_path = output_dir / 'failure_analysis_status.json'
    if not status_path.exists():
        write_json(status_path, {'status': 'pending_human_ratings_and_case_reviews',
                                 'automatic_stage_diagnostics_complete': True,
                                 'artifact_fingerprint': artifacts['fingerprint']})


def select_analysis_cases(records):
    """Sugestão reproduzível; pesos iguais nos três critérios aplicáveis a A e B.

    O escore auxiliar não é exigência do professor. Rastreabilidade é excluída
    desse delta, pois A não tem nota aplicável. Categorias ausentes não são forçadas.
    """
    by_query = {(r['query_id'], r['condition']): r for r in records}
    scores = []
    for qid in RAG_QUERY_IDS:
        a, b = by_query[qid, 'A'], by_query[qid, 'B']
        if any(r[c] is None for r in (a, b) for c in COMMON_CRITERIA):
            return {'status': 'pending_human_ratings', 'cases': [], 'comparisons': []}
        delta = {c: b[c] - a[c] for c in COMMON_CRITERIA}
        scores.append({'query_id': qid, 'mean_delta_common': statistics.mean(delta.values()),
                       'mean_A_common': statistics.mean(a[c] for c in COMMON_CRITERIA),
                       'mean_B_common': statistics.mean(b[c] for c in COMMON_CRITERIA),
                       **{f'delta_{c}': v for c, v in delta.items()}})
    selected, used = [], set()
    improvements = [r for r in scores if r['mean_delta_common'] > 0]
    if improvements:
        chosen = sorted(improvements, key=lambda r: (-r['mean_delta_common'], r['query_id']))[0]
        selected.append({'category': 'melhora', **chosen}); used.add(chosen['query_id'])
    failures = [r for r in scores if r['query_id'] not in used and r['mean_delta_common'] <= 0]
    if failures:
        chosen = sorted(failures, key=lambda r: (r['mean_delta_common'], r['query_id']))[0]
        selected.append({'category': 'nao_ajudou', **chosen}); used.add(chosen['query_id'])
    # No máximo um ponto de mudança total entre os critérios. Evita tratar
    # como empate uma melhora grande em um critério que cancela piora em outro.
    neutral = [r for r in scores if r['query_id'] not in used
               and sum(abs(r[f'delta_{c}']) for c in COMMON_CRITERIA) <= 1]
    if neutral:
        chosen = sorted(neutral, key=lambda r: (abs(r['mean_delta_common']), r['query_id']))[0]
        selected.append({'category': 'pouca_diferenca', **chosen})
    missing = [c for c in CATEGORIES if c not in {r['category'] for r in selected}]
    return {'status': 'suggestions_ready' if not missing else 'categories_not_observed',
            'criterion': 'mean(B-A) of fidelidade, completude and utilidade; equal weights',
            'neutral_absolute_delta_limit': 1/3, 'manual_confirmation_required': True,
            'neutral_total_absolute_change_limit': 1,
            'missing_categories': missing, 'cases': selected, 'comparisons': scores}


def finalize_failure_analysis(experiment_dir='Experiments/rag_experiment', data_dir='nfcorpus/nfcorpus',
                               output_dir=None, reviews_path=None):
    """Publica três análises somente com notas humanas e evidências preenchidas."""
    artifacts = load_evaluation_artifacts(experiment_dir, data_dir)
    output_dir = Path(output_dir or Path(experiment_dir) / 'evaluation')
    human_status_path = output_dir / 'human_evaluation_status.json'
    if not human_status_path.exists() or json.loads(human_status_path.read_text()).get('status') != 'complete':
        path = output_dir / 'failure_analysis_status.json'
        write_json(path, {'status': 'pending_human_ratings',
                          'artifact_fingerprint': artifacts['fingerprint'],
                          'automatic_stage_diagnostics_complete': True})
        return path
    human_status = json.loads(human_status_path.read_text())
    if human_status.get('artifact_fingerprint') != artifacts['fingerprint']:
        raise ValueError('As notas consolidadas pertencem a outro experimento.')
    ratings_path = output_dir / 'human_scores_per_answer.csv'
    if hashlib.sha256(ratings_path.read_bytes()).hexdigest() != human_status['ratings_csv_sha256']:
        raise ValueError('Notas consolidadas modificadas; execute novamente a consolidação das fichas.')
    manifest = json.loads((output_dir / 'evaluation_manifest.json').read_text())
    if manifest['artifact_fingerprint'] != artifacts['fingerprint']:
        raise ValueError('O diagnóstico pertence a outro experimento.')
    for name, expected in manifest['diagnostic_sha256'].items():
        if hashlib.sha256((output_dir / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f'Diagnóstico modificado: {name}; regenere o preparo.')
    selection = json.loads((output_dir / 'failure_case_selection.json').read_text())
    if reviews_path is None:
        reviews_path = prepare_case_review_template(artifacts, output_dir, selection)
    review = json.loads(Path(reviews_path).read_text())
    if review.get('artifact_fingerprint') != artifacts['fingerprint']:
        raise ValueError('A revisão dos casos não corresponde às respostas avaliadas.')
    cases = review.get('cases', [])
    if len(cases) != 3 or {c.get('category') for c in cases} != set(CATEGORIES):
        raise ValueError('A análise exige um caso de cada categoria do item 14.')
    pending = not review.get('reviewer_id', '').strip()
    for c in cases:
        pending |= (not all(str(c.get(k, '')).strip() for k in (
            'query_id', 'selection_reason', 'comparison', 'primary_stage', 'stage_reason'))
                    or c.get('information_in_context') == 'pending'
                    or not c.get('evidence') or any(not e.get('doc_id') or not e.get('quote') for e in c['evidence']))
    if pending:
        path = output_dir / 'failure_analysis_status.json'
        write_json(path, {'status': 'pending_human_case_reviews',
                          'artifact_fingerprint': artifacts['fingerprint'],
                          'template': str(reviews_path)})
        return path
    if len({c['query_id'] for c in cases}) != 3:
        raise ValueError('Use três consultas distintas.')
    metrics = pd.read_csv(output_dir / 'stage_metrics_per_query.csv').set_index('query_id')
    stage_docs = pd.read_csv(output_dir / 'stage_document_evidence.csv')
    ratings = pd.read_csv(ratings_path, dtype=str, keep_default_na=False)
    parts = ['# Análise de falhas ponta a ponta — item 14\n',
             'Revisão declarada por: ' + review['reviewer_id'] + '.\n',
             'Seleção final confirmada pelo revisor; as sugestões numéricas usam a média '
             'dos três critérios comuns, com pesos iguais. Rastreabilidade de A é N/A.\n']
    for c in cases:
        qid = c['query_id']
        if qid not in RAG_QUERY_IDS or c['primary_stage'] not in STAGES:
            raise ValueError('Consulta ou etapa inválida na revisão.')
        if c['information_in_context'] not in ('yes', 'no', 'partial'):
            raise ValueError('Informe se a informação necessária chegou ao contexto: yes/no/partial.')
        if c['category'] == 'nao_ajudou' and c['primary_stage'] == 'none':
            raise ValueError('No caso sem ajuda, identifique e justifique o estágio investigado.')
        case = next(x for x in artifacts['bundle']['cases'] if x['query_id'] == qid)
        context_ids = {d['doc_id'] for d in case['contexts']}
        arrived = False
        verified_quotes = []
        for evidence in c['evidence']:
            doc = artifacts['documents'].get(evidence['doc_id'])
            if doc is None or evidence['quote'] not in (doc['title'] + '\n' + doc['text']):
                raise ValueError(f'Trecho inexistente no corpus: {qid}/{evidence["doc_id"]}.')
            sent = evidence['doc_id'] in context_ids
            arrived |= sent
            verified_quotes.append({**evidence, 'provided_to_model': sent})
        if c['information_in_context'] in ('yes', 'partial') and not arrived:
            raise ValueError('A afirmação de chegada da informação precisa de um trecho do contexto efetivo.')
        s = metrics.loc[qid]
        pair = ratings[(ratings.query_id == qid) &
                       (ratings.evaluator_id == human_status['final_reviewer'])].set_index('condition')
        if set(pair.index) != {'A', 'B'} or len(pair) != 2:
            raise ValueError(f'Notas finais ausentes/duplicadas: {qid}.')
        parts.extend([f'## {c["category"]} — {qid}: {case["query"]}\n',
                      '**Motivo da seleção:** ' + c['selection_reason'] + '\n',
                      '**Comparação das respostas:** ' + c['comparison'] + '\n',
                      f'**Informação necessária no contexto:** {c["information_in_context"]}.\n',
                      f'**Estágio principal:** {c["primary_stage"]}. {c["stage_reason"]}\n',
                      f'Relevantes na união Top-100: {s.relevant_in_candidate_union}; '
                      f'no Top-50 híbrido: {s.relevant_in_hybrid_top50}; '
                      f'no Top-5 híbrido/final: {s.relevant_in_hybrid_top5}/{s.relevant_in_context}.\n',
                      '| Critério | Nota A | Nota B |', '|---|---:|---:|'])
        for criterion in (*COMMON_CRITERIA, 'rastreabilidade'):
            parts.append(f'| {criterion} | {pair.loc["A", criterion]} | {pair.loc["B", criterion]} |')
        parts.extend(['', '| Documento | Grau | Lexical | Semântico | RRF | Reranking | Enviado |',
                      '|---|---:|---:|---:|---:|---:|---|'])
        ids = context_ids | {q['doc_id'] for q in verified_quotes}
        for doc_id in sorted(ids):
            group = stage_docs[(stage_docs.query_id == qid) & (stage_docs.doc_id == doc_id)]
            if group.empty:
                positions = ['—'] * 4
            else:
                row = group.iloc[0]
                positions = ['—' if pd.isna(row[f'{stage}_rank']) else str(int(row[f'{stage}_rank']))
                             for stage in RANKING_FILES]
            grade = artifacts['judgments'][qid].get(doc_id, 0)
            parts.append(f'| {doc_id} | {grade:g} | ' + ' | '.join(positions) + f' | {doc_id in context_ids} |')
        parts.append('\n### Evidências conferidas no corpus\n')
        for q in verified_quotes:
            parts.append(f'{q["doc_id"]}; enviado à LLM: {q["provided_to_model"]}.\n\n> '
                         + q['quote'].replace('\n', '\n> ') + '\n')
        for condition in ('A', 'B'):
            answer = artifacts['pairs'][qid, condition]
            parts.append(f'### Resposta {condition}\n\n{answer["answer"]}\n')
    path = output_dir / 'failure_analysis.md'
    path.write_text('\n'.join(parts), encoding='utf-8')
    write_json(output_dir / 'failure_analysis_status.json', {
        'status': 'complete', 'artifact_fingerprint': artifacts['fingerprint'],
        'ratings_fingerprint': human_status['ratings_fingerprint'],
        'reviewer_id': review['reviewer_id'], 'query_ids': [c['query_id'] for c in cases],
        'source_reviews': str(reviews_path), 'report': str(path)})
    return path
