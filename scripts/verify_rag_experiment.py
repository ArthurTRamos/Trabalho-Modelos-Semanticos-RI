"""Confere os vinte resultados reais do item 12 e resume custos de geração.

Não atribui notas de qualidade às respostas: essas notas exigem avaliação humana.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import statistics


def verify(output_dir, data_dir):
    output_dir, data_dir = Path(output_dir), Path(data_dir)
    status = json.loads((output_dir / 'run_status.json').read_text())
    assert status['status'] == 'complete' and status['completed_responses'] == 20
    bundle = json.loads((output_dir / 'contexts.json').read_text())
    documents = {d['_id']: d for d in map(json.loads, (data_dir / 'corpus.jsonl').read_text().splitlines())}
    for name, expected in bundle['dataset_sha256'].items():
        assert hashlib.sha256((data_dir / name).read_bytes()).hexdigest() == expected
    rows = [json.loads(line) for line in (output_dir / 'answers.jsonl').read_text().splitlines()]
    assert len(rows) == 20
    with (output_dir / 'answers.csv').open(newline='') as file:
        assert len(list(csv.DictReader(file))) == 20
    with (output_dir / 'rag_experiment.csv').open(newline='') as file:
        standardized = list(csv.DictReader(file))
    assert len(standardized) == 20
    assert [r['condition'] for r in standardized] == ['A', 'B'] * 10
    assert all(key in standardized[0] for key in (
        'context_doc_ids', 'prompt', 'answer', 'fidelidade', 'completude', 'utilidade',
        'rastreabilidade', 'justificativa_fidelidade', 'justificativa_completude',
        'justificativa_utilidade', 'justificativa_rastreabilidade',
    ))
    for index, case in enumerate(bundle['cases']):
        a, b = rows[index * 2:index * 2 + 2]
        assert a['condition'] == 'A_without_rag' and b['condition'] == 'B_with_rag'
        assert a['query_id'] == b['query_id'] == case['query_id']
        assert a['query'] == b['query'] == case['query']
        assert a['manifest'] == b['manifest'] == status['manifest']
        assert a['generation_kwargs'] == b['generation_kwargs']
        assert a['experiment_signature'] == b['experiment_signature'] == status['experiment_signature']
        assert a['external_context_provided'] is False and a['contexts'] == []
        assert a['messages'][1]['content'] == case['query']
        assert b['external_context_provided'] is True and b['contexts'] == case['contexts']
        assert len(b['contexts']) == len({d['doc_id'] for d in b['contexts']}) == 5
        assert [d['doc_id'] for d in b['contexts']] == [d['doc_id'] for d in case['reranked_top100'][:5]]
        for d in b['contexts']:
            assert d['title'] == documents[d['doc_id']]['title']
            assert d['text'] == documents[d['doc_id']]['text']
            assert d['hybrid_rank'] <= 50
        for row in (a, b):
            assert row['answer'].strip()
            config = row['manifest']['config']
            assert 0 < row['output_tokens'] <= config['max_new_tokens']
            assert row['input_tokens'] + config['max_new_tokens'] <= row['manifest']['context_window_tokens']
            assert row['messages'][0]['content'].startswith('Answer in English.')
    report = {'verified': True, 'response_count': 20, 'query_count': 10,
              'identical_model_and_parameters_in_each_pair': True,
              'full_context_matches_corpus': True, 'human_evaluation': 'not_assessed_by_this_script',
              'generation_summary': {}}
    for condition in ('A_without_rag', 'B_with_rag'):
        selected = [r for r in rows if r['condition'] == condition]
        report['generation_summary'][condition] = {
            'responses': len(selected),
            'total_generation_seconds': sum(r['generation_seconds'] for r in selected),
            'mean_generation_seconds': statistics.mean(r['generation_seconds'] for r in selected),
            'mean_input_tokens': statistics.mean(r['input_tokens'] for r in selected),
            'mean_output_tokens': statistics.mean(r['output_tokens'] for r in selected),
            'max_input_tokens': max(r['input_tokens'] for r in selected),
            'responses_at_output_token_limit': [r['query_id'] for r in selected
                                               if r['output_tokens'] == r['manifest']['config']['max_new_tokens']],
        }
    (output_dir / 'verification.json').write_text(json.dumps(report, ensure_ascii=False, indent=2))
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', default='Experiments/rag_experiment')
    parser.add_argument('--data-dir', default='nfcorpus/nfcorpus')
    args = parser.parse_args()
    print(json.dumps(verify(args.output_dir, args.data_dir), ensure_ascii=False, indent=2))
