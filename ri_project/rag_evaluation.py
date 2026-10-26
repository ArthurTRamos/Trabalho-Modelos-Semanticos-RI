"""Item 13: métricas dos contextos e avaliação humana auditável, sem inferência."""
import argparse
import csv
import hashlib
import itertools
import json
from pathlib import Path
import random
import statistics

import pandas as pd

from .data import load_dataset
from .rag_experiment import RAG_QUERY_IDS, export_standardized_csv

CRITERIA = ('fidelidade', 'completude', 'utilidade', 'rastreabilidade')
COMMON_CRITERIA = CRITERIA[:3]
CONDITIONS = {'A_without_rag': 'A', 'B_with_rag': 'B'}
RANKING_FILES = {'lexical': 'results_lexical.csv', 'semantic': 'results_semantic.csv',
                 'hybrid': 'results_hybrid.csv', 'reranking': 'results_reranking.csv'}


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    temporary.replace(path)


def artifact_fingerprint(bundle, answers):
    raw = json.dumps({'bundle': bundle, 'answers': answers}, sort_keys=True,
                     ensure_ascii=False).encode()
    return hashlib.sha256(raw).hexdigest()


def load_evaluation_artifacts(experiment_dir='Experiments/rag_experiment',
                              data_dir='nfcorpus/nfcorpus'):
    """Confere a correspondência entre respostas, contexto real e dataset."""
    experiment_dir, data_dir = Path(experiment_dir), Path(data_dir)
    status = json.loads((experiment_dir / 'run_status.json').read_text())
    if status.get('status') != 'complete' or status.get('completed_responses') != 20:
        raise ValueError('Conclua as vinte respostas do item 12 antes de avaliar.')
    bundle = json.loads((experiment_dir / 'contexts.json').read_text())
    if bundle.get('query_ids') != list(RAG_QUERY_IDS):
        raise ValueError('O contexto não corresponde às dez consultas oficiais.')
    expected_files = ('corpus.jsonl', 'queries.jsonl', 'qrels/dev.tsv', 'qrels/test.tsv')
    for name in expected_files:
        digest = hashlib.sha256((data_dir / name).read_bytes()).hexdigest()
        if bundle.get('dataset_sha256', {}).get(name) != digest:
            raise ValueError(f'Dataset divergente ou sem hash registrado: {name}.')
    answers = [json.loads(line) for line in (experiment_dir / 'answers.jsonl').read_text().splitlines() if line.strip()]
    pairs = {(r['query_id'], CONDITIONS[r['condition']]): r for r in answers}
    expected = {(q, c) for q in RAG_QUERY_IDS for c in ('A', 'B')}
    if len(answers) != 20 or set(pairs) != expected:
        raise ValueError('As respostas precisam conter dez pares únicos A/B.')
    cases = {c['query_id']: c for c in bundle['cases']}
    if len(bundle['cases']) != 10 or set(cases) != set(RAG_QUERY_IDS):
        raise ValueError('Contextos ausentes ou duplicados.')
    dataset = load_dataset(data_dir)
    judgments = {}
    for qid, doc_id, grade in dataset['qrels']['test'].itertuples(index=False, name=None):
        group = judgments.setdefault(qid, {})
        group[doc_id] = max(float(grade), group.get(doc_id, 0.0))
    for qid, case in cases.items():
        a, b = pairs[qid, 'A'], pairs[qid, 'B']
        if (a['contexts'] or a['external_context_provided'] is not False
                or b['external_context_provided'] is not True
                or a['manifest'] != b['manifest']
                or a['generation_kwargs'] != b['generation_kwargs']):
            raise ValueError(f'Pareamento/isolamento A/B inválido: {qid}.')
        if any(r['experiment_signature'] != status['experiment_signature'] for r in (a, b)):
            raise ValueError(f'Assinatura experimental divergente: {qid}.')
        if b['contexts'] != case['contexts'] or len(case['contexts']) != 5:
            raise ValueError(f'Contexto B não coincide com o pacote: {qid}.')
        ids = [d['doc_id'] for d in case['contexts']]
        if len(set(ids)) != 5 or ids != [d['doc_id'] for d in case['reranked_top100'][:5]]:
            raise ValueError(f'Top-5 inválido: {qid}.')
        for d in case['contexts']:
            original = dataset['documents'][d['doc_id']]
            if d['title'] != original['title'] or d['text'] != original['text']:
                raise ValueError(f'Documento alterado: {qid}/{d["doc_id"]}.')
        for answer in (a, b):
            if answer['query'] != case['query'] or not answer['answer'].strip():
                raise ValueError(f'Texto de consulta/resposta inválido: {qid}.')
    return {'bundle': bundle, 'answers': answers, 'pairs': pairs,
            'documents': dataset['documents'], 'judgments': judgments,
            'fingerprint': artifact_fingerprint(bundle, answers)}


def context_metrics(doc_ids, relevance, k=5):
    """Métricas binárias; recall usa todos os relevantes nos qrels."""
    if len(doc_ids) != k or len(set(doc_ids)) != k:
        raise ValueError(f'Exija exatamente {k} documentos únicos no contexto.')
    total = sum(grade > 0 for grade in relevance.values())
    hits = sum(relevance.get(doc_id, 0.0) > 0 for doc_id in doc_ids)
    return {'relevant_in_context': hits, 'total_relevant': total,
            'P5': hits / k, 'R5': hits / total if total else 0.0,
            'Hit5': int(hits > 0)}


def export_context_metrics(artifacts, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for case in artifacts['bundle']['cases']:
        ids = [d['doc_id'] for d in case['contexts']]
        rows.append({'query_id': case['query_id'], 'query': case['query'],
                     'context_doc_ids': json.dumps(ids),
                     **context_metrics(ids, artifacts['judgments'][case['query_id']])})
    per_query = pd.DataFrame(rows)
    summary = pd.DataFrame([{'split': 'test', 'query_count': len(rows),
                             **{m: per_query[m].mean() for m in ('P5', 'R5', 'Hit5')}}])
    per_query.to_csv(output_dir / 'context_metrics_per_query.csv', index=False)
    summary.to_csv(output_dir / 'context_metrics_summary.csv', index=False)
    return per_query, summary


def prepare_human_review(artifacts, output_dir, seed=42):
    """Cria fichas sem rótulos de condição; nunca sobrescreve notas preenchidas."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    key_path = output_dir / 'blind_key.json'
    if key_path.exists():
        key = json.loads(key_path.read_text())
        if key['artifact_fingerprint'] != artifacts['fingerprint'] or key['shuffle_seed'] != seed:
            raise ValueError('Respostas/contextos/seed mudaram. Use outra pasta de avaliação para preservar as notas.')
    else:
        ordered = list(artifacts['answers'])
        random.Random(seed).shuffle(ordered)
        key = {'artifact_fingerprint': artifacts['fingerprint'], 'shuffle_seed': seed,
               'items': [{'answer_id': f'R{i:02}', 'query_id': r['query_id'],
                          'condition': CONDITIONS[r['condition']]}
                         for i, r in enumerate(ordered, 1)]}
        write_json(key_path, key)
    rubric = '''# Avaliação humana — item 13

Preencha uma ficha CSV por integrante, de forma independente. Não consulte blind_key.json ou os rankings antes de concluir as notas. A ordem das vinte respostas foi embaralhada com seed 42; rótulos A/B não constam deste caderno. O estilo de escrita ou as citações podem sugerir a condição: o mascaramento não garante cegamento completo.

Use 0, 1 ou 2 e uma justificativa breve por critério. Campos vazios significam pendência e não serão convertidos em zero. Compare a fidelidade de TODAS as respostas com os cinco documentos de referência abaixo; a mesma referência é usada nos dois textos de uma consulta. Para uma das condições a LLM não recebeu contexto externo; isso será explicitado ao revelar a chave.

| Critério | 0 | 1 | 2 |
|---|---|---|---|
| Fidelidade | Afirmações importantes sem suporte ou contraditórias | Parcialmente sustentada | Principais afirmações sustentadas |
| Completude | Não responde aos aspectos principais | Responde parcialmente | Cobre os aspectos respondíveis |
| Utilidade | Não atende à consulta | Parcialmente útil | Atende diretamente à consulta |
| Rastreabilidade | Fontes não identificáveis | Apenas parte das afirmações associável às fontes | Principais afirmações associáveis aos doc_ids |

Na etapa mascarada, avalie rastreabilidade provisoriamente para todas as respostas, inclusive com 0 quando faltam fontes. Na consolidação, as notas provisórias da condição A serão descartadas e registradas como N/A, fora da média. A condição B exige nota e justificativa. Não interprete relevância dos qrels como garantia de suporte para uma afirmação específica. Edite somente sua ficha, mantendo answer_id e evaluator_id.
'''
    parts = [rubric]
    for item in key['items']:
        row = artifacts['pairs'][item['query_id'], item['condition']]
        case = next(c for c in artifacts['bundle']['cases'] if c['query_id'] == item['query_id'])
        parts.extend([f'\n## {item["answer_id"]} — {row["query"]}\n',
                      '### Resposta a avaliar\n\n' + row['answer'] + '\n',
                      '### Documentos de referência para avaliação\n'])
        for d in case['contexts']:
            parts.append(f'#### {d["doc_id"]} — {d["title"]}\n\n{d["text"]}\n')
    (output_dir / 'human_review.md').write_text('\n'.join(parts), encoding='utf-8')
    columns = ['answer_id', 'evaluator_id'] + [x for c in CRITERIA for x in (c, f'justificativa_{c}')]
    for number in (1, 2):
        path = output_dir / f'ratings_reviewer_{number}.csv'
        if path.exists():
            continue
        with path.open('w', encoding='utf-8', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=columns)
            writer.writeheader()
            for item in key['items']:
                writer.writerow({'answer_id': item['answer_id'], 'evaluator_id': f'avaliador_{number}'})
    return key


def _score(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise ValueError(f'Nota inválida: {value!r}. Use 0, 1 ou 2.') from None
    if number not in (0, 1, 2):
        raise ValueError(f'Nota inválida: {value!r}. Use 0, 1 ou 2.')
    return int(number)


def read_human_ratings(paths, key):
    """Valida IDs, notas, justificativas e N/A; permite rascunhos incompletos."""
    lookup = {r['answer_id']: r for r in key['items']}
    result, seen = [], set()
    for path in paths:
        with Path(path).open(encoding='utf-8-sig', newline='') as file:
            reader = csv.DictReader(file)
            required = {'answer_id', 'evaluator_id'} | {
                field for c in CRITERIA for field in (c, f'justificativa_{c}')}
            if not required.issubset(reader.fieldnames or []):
                raise ValueError(f'{path}: cabeçalho da ficha incompleto.')
            rows = list(reader)
        if len(rows) != 20 or {r['answer_id'] for r in rows} != set(lookup):
            raise ValueError(f'{path}: mantenha exatamente os vinte answer_id do formulário.')
        evaluators = {r.get('evaluator_id', '').strip() for r in rows}
        if len(evaluators) != 1 or not next(iter(evaluators)):
            raise ValueError(f'{path}: use um único evaluator_id não vazio por arquivo.')
        evaluator = next(iter(evaluators))
        for row in rows:
            item = lookup[row['answer_id']]
            identity = evaluator, row['answer_id']
            if identity in seen:
                raise ValueError(f'Avaliação duplicada: {identity}.')
            seen.add(identity)
            record = {**item, 'evaluator_id': evaluator}
            for criterion in CRITERIA:
                raw = (row.get(criterion) or '').strip()
                reason = (row.get(f'justificativa_{criterion}') or '').strip()
                if item['condition'] == 'A' and criterion == 'rastreabilidade':
                    if raw and raw != 'N/A':
                        _score(raw)
                    record[criterion] = None
                    record[f'justificativa_{criterion}'] = 'N/A: nenhum contexto externo fornecido à LLM.'
                    continue
                record[criterion] = _score(raw) if raw else None
                if raw and not reason:
                    raise ValueError(f'{path}/{row["answer_id"]}: falta justificativa de {criterion}.')
                record[f'justificativa_{criterion}'] = reason
            result.append(record)
    if not result:
        raise ValueError('Informe ao menos uma ficha de avaliação.')
    return result


def agreement_statistics(left, right):
    """Concordância exata e kappa de Cohen com pesos quadráticos para 0–2."""
    if len(left) != len(right) or not left:
        return {'n_pairs': 0, 'exact_agreement': None, 'quadratic_kappa': None}
    n = len(left)
    observed = sum((a - b) ** 2 / 4 for a, b in zip(left, right)) / n
    expected = sum(left.count(a) * right.count(b) * (a - b) ** 2 / 4
                   for a in (0, 1, 2) for b in (0, 1, 2)) / n**2
    return {'n_pairs': n, 'exact_agreement': sum(a == b for a, b in zip(left, right)) / n,
            'quadratic_kappa': 1 - observed / expected if expected else None}


def summarize_human_ratings(records):
    rows, agreements = [], []
    evaluators = sorted({r['evaluator_id'] for r in records})
    for evaluator in evaluators:
        for condition in ('A', 'B'):
            group = [r for r in records if r['evaluator_id'] == evaluator and r['condition'] == condition]
            for criterion in CRITERIA:
                applicable = not (condition == 'A' and criterion == 'rastreabilidade')
                scores = [r[criterion] for r in group if r[criterion] is not None] if applicable else []
                rows.append({'evaluator_id': evaluator, 'condition': condition, 'criterion': criterion,
                             'applicable': applicable, 'expected_count': 10 if applicable else 0,
                             'rated_count': len(scores), 'mean': statistics.mean(scores) if scores else None,
                             'complete': len(scores) == (10 if applicable else 0)})
    for a, b in itertools.combinations(evaluators, 2):
        for condition in ('A', 'B'):
            for criterion in CRITERIA:
                if condition == 'A' and criterion == 'rastreabilidade':
                    continue
                left = {r['answer_id']: r[criterion] for r in records
                        if r['evaluator_id'] == a and r['condition'] == condition}
                right = {r['answer_id']: r[criterion] for r in records
                         if r['evaluator_id'] == b and r['condition'] == condition}
                ids = sorted(k for k in left if left[k] is not None and right.get(k) is not None)
                agreements.append({'evaluator_a': a, 'evaluator_b': b, 'condition': condition,
                                   'criterion': criterion, **agreement_statistics(
                                       [left[k] for k in ids], [right[k] for k in ids])})
    return pd.DataFrame(rows), pd.DataFrame(agreements, columns=[
        'evaluator_a', 'evaluator_b', 'condition', 'criterion', 'n_pairs',
        'exact_agreement', 'quadratic_kappa'])


def write_human_report(summary, status, selection, output_dir):
    """Discute diferenças numéricas somente quando há notas humanas completas."""
    output_dir = Path(output_dir)
    comparisons = pd.DataFrame(selection.get('comparisons', []), columns=[
        'query_id', 'mean_A_common', 'mean_B_common', 'mean_delta_common',
        *[f'delta_{c}' for c in COMMON_CRITERIA]])
    comparisons.to_csv(output_dir / 'quality_comparison_per_query.csv', index=False)
    parts = ['# Avaliação da geração — item 13\n',
             f'Estado: {status["status"]}.\n',
             'A não recebeu contexto externo. Sua fidelidade é comparada com os mesmos '
             'Top-5 enviados em B; rastreabilidade de A é N/A e não participa da média. '
             'Campos vazios nas fichas significam pendência, não nota zero.\n']
    if status['status'] != 'complete':
        parts.append('Não há comparação final de qualidade. Preencha as fichas e, havendo '
                     'mais de um avaliador completo, escolha explicitamente o responsável '
                     'pelas notas finais. Médias parciais e suas contagens estão em human_scores_summary.csv.\n')
    else:
        selected = summary[summary.evaluator_id == status['final_reviewer']]
        parts.extend([f'Avaliador final: {status["final_reviewer"]}.\n',
                      '| Critério | Média A | Média B | Δ B − A |',
                      '|---|---:|---:|---:|'])
        for criterion in CRITERIA:
            a = selected[(selected.condition == 'A') & (selected.criterion == criterion)].iloc[0]['mean']
            b = selected[(selected.condition == 'B') & (selected.criterion == criterion)].iloc[0]['mean']
            if criterion == 'rastreabilidade':
                parts.append(f'| {criterion} | N/A | {b:.4f} | N/A |')
            else:
                parts.append(f'| {criterion} | {a:.4f} | {b:.4f} | {b-a:+.4f} |')
        delta = comparisons.mean_delta_common
        parts.extend(['', f'Na média auxiliar dos três critérios comuns, B supera A em '
                      f'{int((delta > 0).sum())} consultas, empata em {int((delta == 0).sum())} '
                      f'e fica abaixo em {int((delta < 0).sum())}. A média auxiliar usa pesos '
                      'iguais e serve à sugestão dos casos do item 14, sem substituir as '
                      'quatro médias ou as justificativas individuais.\n',
                      'A ausência de fontes em A limita a verificação das afirmações pelo '
                      'leitor. A nota de rastreabilidade de B mede associação das afirmações '
                      'a doc_ids; a mera presença de uma citação não garante fidelidade.\n'])
    parts.append('A concordância entre avaliadores está em human_agreement.csv, com '
                 'número de pares, concordância exata e kappa de Cohen com pesos quadráticos. '
                 'Kappa sem variabilidade marginal é indefinido e permanece vazio; '
                 'resultados com avaliações parciais são provisórios.')
    (output_dir / 'human_evaluation_report.md').write_text('\n'.join(parts) + '\n', encoding='utf-8')


def aggregate_human_evaluation(paths, experiment_dir='Experiments/rag_experiment',
                               data_dir='nfcorpus/nfcorpus', output_dir=None,
                               final_reviewer=None):
    artifacts = load_evaluation_artifacts(experiment_dir, data_dir)
    output_dir = Path(output_dir or Path(experiment_dir) / 'evaluation')
    key = json.loads((output_dir / 'blind_key.json').read_text())
    if key['artifact_fingerprint'] != artifacts['fingerprint']:
        raise ValueError('As fichas pertencem a outras respostas/contextos.')
    records = read_human_ratings(paths, key)
    summary, agreement = summarize_human_ratings(records)
    summary.to_csv(output_dir / 'human_scores_summary.csv', index=False)
    agreement.to_csv(output_dir / 'human_agreement.csv', index=False)
    pd.DataFrame([{**r, 'rastreabilidade': 'N/A'} if r['condition'] == 'A' else r
                  for r in records]).to_csv(output_dir / 'human_scores_per_answer.csv', index=False)
    complete = [name for name, g in summary.groupby('evaluator_id') if g.complete.all()]
    if final_reviewer is not None and final_reviewer not in set(summary.evaluator_id):
        raise ValueError('final_reviewer não consta das fichas informadas.')
    selected = final_reviewer or (complete[0] if len(complete) == 1 else None)
    done = selected in complete if selected is not None else False
    rated = int(summary.rated_count.sum())
    ratings_fingerprint = hashlib.sha256(json.dumps(
        {'records': sorted(records, key=lambda r: (r['evaluator_id'], r['answer_id'])),
         'selected': selected}, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    status = {'status': 'complete' if done else ('needs_final_reviewer' if len(complete) > 1 and selected is None
                                               else 'partial' if rated else 'pending_human_ratings'),
              'artifact_fingerprint': artifacts['fingerprint'], 'final_reviewer': selected,
              'ratings_fingerprint': ratings_fingerprint,
              'ratings_csv_sha256': hashlib.sha256((output_dir / 'human_scores_per_answer.csv').read_bytes()).hexdigest(),
              'complete_reviewers': complete, 'required_cells_per_reviewer': 70,
              'rated_cells': rated, 'quality_scores_created_automatically': False}
    write_json(output_dir / 'human_evaluation_status.json', status)
    analysis_status = output_dir / 'failure_analysis_status.json'
    previous_analysis = json.loads(analysis_status.read_text()) if analysis_status.exists() else {}
    if previous_analysis.get('ratings_fingerprint') != ratings_fingerprint:
        if (output_dir / 'failure_analysis.md').exists():
            old_hash = previous_analysis.get('ratings_fingerprint', 'previous')[:12]
            (output_dir / 'failure_analysis.md').replace(output_dir / f'failure_analysis_{old_hash}.md')
        write_json(analysis_status, {'status': 'pending_human_case_reviews' if done else 'pending_human_ratings',
                                     'artifact_fingerprint': artifacts['fingerprint'],
                                     'ratings_fingerprint': ratings_fingerprint,
                                     'automatic_stage_diagnostics_complete': True})
    if done:
        chosen = {(r['query_id'], r['condition']): r for r in records if r['evaluator_id'] == selected}
        destination = Path(experiment_dir) / 'rag_experiment.csv'
        export_standardized_csv(artifacts['answers'], destination)
        frame = pd.read_csv(destination, keep_default_na=False, dtype=str)
        for index, row in frame.iterrows():
            grade = chosen[row.query_id, row.condition]
            for criterion in CRITERIA:
                frame.loc[index, criterion] = ('N/A' if row.condition == 'A' and criterion == 'rastreabilidade'
                                               else str(grade[criterion]))
                frame.loc[index, f'justificativa_{criterion}'] = grade[f'justificativa_{criterion}']
        frame.to_csv(destination, index=False)
        from .rag_analysis import select_analysis_cases, prepare_case_review_template
        selection = select_analysis_cases([r for r in records if r['evaluator_id'] == selected])
        write_json(output_dir / 'failure_case_selection.json', selection)
        prepare_case_review_template(artifacts, output_dir, selection)
    else:
        selection = {
            'status': 'pending_human_ratings', 'cases': [],
            'reason': 'A seleção final depende de uma avaliação humana completa escolhida.'}
        write_json(output_dir / 'failure_case_selection.json', selection)
    write_human_report(summary, status, selection, output_dir)
    return {'status': status, 'summary': summary, 'agreement': agreement, 'records': records}


def prepare_rag_evaluation(experiment_dir='Experiments/rag_experiment', data_dir='nfcorpus/nfcorpus',
                           rankings_dir='Experiments', output_dir=None):
    artifacts = load_evaluation_artifacts(experiment_dir, data_dir)
    output_dir = Path(output_dir or Path(experiment_dir) / 'evaluation')
    per_query, summary = export_context_metrics(artifacts, output_dir)
    prepare_human_review(artifacts, output_dir)
    from .rag_analysis import build_stage_diagnostics, write_analysis_dossiers
    diagnostics = build_stage_diagnostics(artifacts, rankings_dir, output_dir)
    write_analysis_dossiers(artifacts, diagnostics, output_dir)
    write_json(output_dir / 'evaluation_manifest.json', {
        'artifact_fingerprint': artifacts['fingerprint'], 'query_ids': list(RAG_QUERY_IDS),
        'split': 'test', 'k': 5, 'relevant': 'qrel > 0', 'macro_average': True,
        'human_scores_created_automatically': False,
        'ranking_sha256': {name: hashlib.sha256((Path(rankings_dir) / name).read_bytes()).hexdigest()
                           for name in RANKING_FILES.values()},
        'diagnostic_sha256': {name: hashlib.sha256((output_dir / name).read_bytes()).hexdigest()
                              for name in ('stage_metrics_per_query.csv', 'stage_document_evidence.csv')}})
    return {'context_metrics': per_query, 'context_summary': summary,
            'diagnostics': diagnostics, 'output_dir': output_dir}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('prepare', 'aggregate', 'analyze'))
    parser.add_argument('--experiment-dir', default='Experiments/rag_experiment')
    parser.add_argument('--data-dir', default='nfcorpus/nfcorpus')
    parser.add_argument('--rankings-dir', default='Experiments')
    parser.add_argument('--output-dir')
    parser.add_argument('--ratings', nargs='+', type=Path)
    parser.add_argument('--final-reviewer')
    parser.add_argument('--case-reviews', type=Path)
    args = parser.parse_args()
    output_dir = Path(args.output_dir or Path(args.experiment_dir) / 'evaluation')
    if args.stage == 'prepare':
        result = prepare_rag_evaluation(args.experiment_dir, args.data_dir,
                                        args.rankings_dir, output_dir)
        print(result['context_summary'].to_string(index=False))
    elif args.stage == 'aggregate':
        if not args.ratings:
            parser.error('Informe --ratings com as fichas preenchidas.')
        result = aggregate_human_evaluation(args.ratings, args.experiment_dir,
                                            args.data_dir, output_dir, args.final_reviewer)
        print(json.dumps(result['status'], ensure_ascii=False, indent=2))
        print(result['summary'].to_string(index=False))
    else:
        from .rag_analysis import finalize_failure_analysis
        path = finalize_failure_analysis(args.experiment_dir, args.data_dir, output_dir,
                                          args.case_reviews)
        print(path)


if __name__ == '__main__':
    main()
