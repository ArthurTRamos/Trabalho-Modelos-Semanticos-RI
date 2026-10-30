"""Consolida notas explícitas de IA sem apresentá-las como avaliação humana.

As notas são fornecidas em assessments.json, após leitura das respostas/fontes.
Este script valida e agrega esses julgamentos; não atribui notas por heurística.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ri_project.rag_evaluation import (  # noqa: E402
    CRITERIA, aggregate_human_evaluation, prepare_rag_evaluation, write_json,
)
from ri_project.rag_analysis import finalize_failure_analysis  # noqa: E402

NOTICE = (
    '> **SIMULAÇÃO POR IA — NÃO É AVALIAÇÃO HUMANA.** Duas passagens do mesmo '
    'agente, sem independência ou cegamento. O item 13 continua exigindo '
    'validação humana. Nomes de arquivos com `human_` pertencem à infraestrutura '
    'reutilizada e não indicam origem humana das notas.\n\n'
)


def consolidate(source_dir, work_dir, data_dir, rankings_dir):
    """Preserva a rodada original e publica resultados em uma cópia identificada."""
    source_dir, work_dir = Path(source_dir), Path(work_dir)
    if source_dir.resolve() == work_dir.resolve():
        raise ValueError('A simulação precisa de diretório próprio.')
    assessment = json.loads((work_dir / 'assessments.json').read_text(encoding='utf-8'))
    if (assessment.get('evaluation_origin') != 'ai_simulation'
            or assessment.get('human_requirement_satisfied') is not False
            or assessment.get('independent_human_reviewers') is not False):
        raise ValueError('Declare explicitamente a origem IA e a ausência de revisores humanos.')
    for name, expected in assessment['source_sha256'].items():
        actual = hashlib.sha256((source_dir / name).read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f'As notas pertencem a outra rodada: {name}.')
    experiment = work_dir / 'experiment'
    evaluation = work_dir / 'evaluation'
    experiment.mkdir(parents=True, exist_ok=True)
    # Copia apenas as entradas necessárias; notas originais nunca são alteradas.
    for name in ('contexts.json', 'answers.jsonl', 'answers.csv', 'run_status.json'):
        shutil.copy2(source_dir / name, experiment / name)
    prepare_rag_evaluation(experiment, data_dir, rankings_dir, evaluation)
    key = json.loads((evaluation / 'blind_key.json').read_text(encoding='utf-8'))
    columns = ['answer_id', 'evaluator_id', 'evaluation_origin']
    columns += [field for criterion in CRITERIA
                for field in (criterion, f'justificativa_{criterion}')]
    paths = []
    for index, review in enumerate(assessment['reviews'], 1):
        if not review['evaluator_id'].startswith('ia_'):
            raise ValueError('Identifique a origem IA no evaluator_id.')
        by_pair = {(row['query_id'], row['condition']): row for row in review['rows']}
        if len(by_pair) != 20 or len(review['rows']) != 20:
            raise ValueError('Cada passagem deve avaliar os vinte pares query_id/condition únicos.')
        path = evaluation / f'ratings_reviewer_{index}.csv'
        with path.open('w', encoding='utf-8', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=columns)
            writer.writeheader()
            for item in key['items']:
                grade = by_pair[item['query_id'], item['condition']]
                row = {'answer_id': item['answer_id'], 'evaluator_id': review['evaluator_id'],
                       'evaluation_origin': 'ai_simulation'}
                for criterion in CRITERIA:
                    score = grade['scores'][criterion]
                    applicable = not (item['condition'] == 'A' and criterion == 'rastreabilidade')
                    if applicable and (type(score) is not int or score not in (0, 1, 2)):
                        raise ValueError(f'Nota aplicável inválida: {item}/{criterion}.')
                    reason = grade['justifications'][criterion]
                    if not reason.strip():
                        raise ValueError(f'Falta justificativa: {item}/{criterion}.')
                    row[criterion] = score if applicable else 'N/A'
                    row[f'justificativa_{criterion}'] = reason
                writer.writerow(row)
        paths.append(path)
    result = aggregate_human_evaluation(
        paths, experiment_dir=experiment, data_dir=data_dir, output_dir=evaluation,
        final_reviewer=assessment['primary_reviewer'],
    )
    reviews = work_dir / 'case_reviews_ai.json'
    analysis = None
    if reviews.exists():
        analysis = finalize_failure_analysis(experiment, data_dir, evaluation, reviews)
    provenance = {key: assessment[key] for key in (
        'evaluation_origin', 'agent', 'independent_human_reviewers', 'blinded',
        'human_requirement_satisfied', 'limitations', 'primary_reviewer',
        'primary_selection_reason', 'source_sha256')}
    provenance['assessment_sha256'] = hashlib.sha256((work_dir / 'assessments.json').read_bytes()).hexdigest()
    provenance['source_experiment'] = str(source_dir)
    provenance['original_human_forms_modified'] = False
    write_json(work_dir / 'evaluation_provenance.json', provenance)
    # Mantém o status de validação do pipeline, declarando a natureza da avaliação.
    for name in ('human_evaluation_status.json', 'failure_analysis_status.json'):
        path = evaluation / name
        status = json.loads(path.read_text(encoding='utf-8'))
        status.update(evaluation_origin='ai_simulation', human_requirement_satisfied=False,
                      independent_human_reviewers=False)
        write_json(path, status)
    for path in evaluation.glob('*.md'):
        content = path.read_text(encoding='utf-8')
        if not content.startswith(NOTICE):
            path.write_text(NOTICE + content, encoding='utf-8')
    standardized = experiment / 'rag_experiment.csv'
    with standardized.open(encoding='utf-8', newline='') as file:
        reader = csv.DictReader(file)
        fields, rows = list(reader.fieldnames), list(reader)
    if 'evaluation_origin' not in fields:
        fields.append('evaluation_origin')
    with standardized.open('w', encoding='utf-8', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows({**row, 'evaluation_origin': 'ai_simulation'} for row in rows)
    print(result['summary'].to_string(index=False))
    print('Análise:', analysis or 'preencha case_reviews_ai.json com os casos/evidências.')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-dir', type=Path, default=Path('Experiments/rag_experiment'))
    parser.add_argument('--work-dir', type=Path, default=Path('Experiments/rag_ai_review'))
    parser.add_argument('--data-dir', type=Path, default=Path('nfcorpus/nfcorpus'))
    parser.add_argument('--rankings-dir', type=Path, default=Path('Experiments'))
    args = parser.parse_args()
    consolidate(args.source_dir, args.work_dir, args.data_dir, args.rankings_dir)
