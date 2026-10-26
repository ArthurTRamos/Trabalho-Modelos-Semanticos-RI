"""Protocolo pareado do item 12; execução por etapas para limitar memória."""
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import csv
import hashlib
import json
from pathlib import Path
import random
import time

from .rag import RAGConfig, LocalCausalGenerator, build_rag_messages, select_context


RAG_QUERY_IDS = (
    'PLAIN-68', 'PLAIN-2450', 'PLAIN-153', 'PLAIN-2850', 'PLAIN-2750',
    'PLAIN-270', 'PLAIN-2470', 'PLAIN-2430', 'PLAIN-238', 'PLAIN-3342',
)


def select_experiment_queries(queries, test_query_ids):
    """Reproduz e confere o sorteio oficial, sem usar graus de relevância."""
    ids = [qid for qid in sorted(set(test_query_ids))
           if queries[qid]['text'].strip().endswith('?')]
    if len(ids) != 42:
        raise ValueError(f'Esperadas 42 perguntas em test; encontradas {len(ids)}.')
    selected = random.Random(42).sample(ids, 10)
    if selected != list(RAG_QUERY_IDS):
        raise ValueError(f'O sorteio difere da lista oficial: {selected}')
    return selected


def build_no_rag_messages(query, language='English'):
    """A recebe somente a pergunta, além da instrução comum de idioma."""
    if not str(query).strip():
        raise ValueError('A consulta não pode ser vazia.')
    return [{'role': 'system', 'content': f'Answer in {language}.'},
            {'role': 'user', 'content': str(query)}]


def _write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    temporary.replace(path)


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True,
                                    ensure_ascii=False).encode()).hexdigest()


def export_standardized_csv(rows, path):
    """Formato do item 15; notas humanas pendentes ficam em branco, não zero."""
    criteria = ('fidelidade', 'completude', 'utilidade', 'rastreabilidade')
    columns = ['query_id', 'query', 'condition', 'context_doc_ids', 'prompt', 'answer']
    columns += [field for criterion in criteria
                for field in (criterion, f'justificativa_{criterion}')]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            with_rag = row['condition'] == 'B_with_rag'
            result = {field: '' for field in columns}
            result.update(
                query_id=row['query_id'], query=row['query'],
                condition='B' if with_rag else 'A',
                context_doc_ids=json.dumps([d['doc_id'] for d in row['contexts']]),
                prompt=row['prompt'], answer=row['answer'],
            )
            if not with_rag:
                result['rastreabilidade'] = 'N/A'
                result['justificativa_rastreabilidade'] = 'Nenhum contexto externo fornecido à LLM.'
            writer.writerow(result)


def prepare_contexts(data_dir, output_dir):
    """Executa recuperação em CPU/GPU, ajustando RRF exclusivamente em dev.

    Execute em processo separado da LLM: ao sair, os modelos de recuperação
    deixam de ocupar RAM/VRAM. Os documentos enviados à LLM ficam no JSON.
    """
    from .data import load_dataset
    from .indexing import build_inverted_index
    from .bm25 import BM25Retriever
    from .semantic import SemanticRetriever
    from .hybrid import HybridRRFRetriever
    from .reranking import CrossEncoderReranker, HybridRerankingRetriever
    from .evaluation import tune_rrf_k

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset = load_dataset(data_dir)
    documents, queries, qrels = (dataset[key] for key in ('documents', 'queries', 'qrels'))
    query_ids = select_experiment_queries(queries, qrels['test']['query-id'])
    print(f'Dataset: {len(documents)} documentos; dez perguntas oficiais confirmadas.', flush=True)
    bm25 = BM25Retriever(build_inverted_index(documents))
    semantic = SemanticRetriever(documents, queries, qrels)
    print('Embeddings prontos; ajustando RRF apenas em dev.', flush=True)
    summary, _, selected_k = tune_rrf_k(
        queries, qrels, bm25, semantic, documents.keys(),
        output_dir=output_dir / 'retrieval_dev',
    )
    hybrid = HybridRRFRetriever(bm25, semantic, k=selected_k,
                               candidate_depth=100, document_ids=documents.keys())
    reranker = CrossEncoderReranker()
    retriever = HybridRerankingRetriever(hybrid, reranker, documents)
    cases = []
    for qid in query_ids:
        query = queries[qid]['text']
        start = time.perf_counter()
        ranking = retriever.search(query, top_k=100)
        contexts = select_context(ranking, documents, count=5)
        cases.append({'query_id': qid, 'query': query, 'contexts': contexts,
                      'reranked_top100': ranking,
                      'retrieval_and_reranking_seconds': time.perf_counter() - start})
        print(f'Contexto pronto: {qid}', flush=True)
    hashes = {str(name): hashlib.sha256((Path(data_dir) / name).read_bytes()).hexdigest()
              for name in ('corpus.jsonl', 'queries.jsonl', 'qrels/dev.tsv', 'qrels/test.tsv')}
    bundle = {
        'protocol': 'assignment_item_12', 'split': 'test', 'sampling_seed': 42,
        'question_candidate_count': 42, 'query_ids': query_ids,
        'dataset_sha256': hashes, 'cases': cases,
        'retrieval': {
            'bm25': {'k1': bm25.k1, 'b': bm25.b},
            'semantic': semantic.manifest,
            'rrf_k': selected_k, 'rrf_dev_tuning': summary.to_dict('records'),
            'candidate_depth': 100, 'rrf_weights': hybrid.weights,
            'reranker': reranker.manifest(),
            'context_documents': 5, 'context_unit': 'title + full text; one corpus document',
            'context_truncation': False,
        },
    }
    _write_json(output_dir / 'contexts.json', bundle)
    return bundle


def run_paired_experiment(bundle, generator, output_dir):
    """Gera A/B com pesos e parâmetros idênticos, chamadas independentes.

    Checkpoints só são reutilizados quando contexto, prompt e manifesto são
    idênticos. Falha/interrupção deixa o estado como incompleto.
    """
    if bundle['query_ids'] != list(RAG_QUERY_IDS):
        raise ValueError('Use as dez consultas oficiais, na ordem do sorteio.')
    cases = bundle['cases']
    if [case['query_id'] for case in cases] != list(RAG_QUERY_IDS):
        raise ValueError('Casos ausentes, duplicados ou fora da ordem oficial.')
    if generator.config.context_documents != 5:
        raise ValueError('O item 12 exige exatamente cinco documentos de contexto.')
    for case in cases:
        if len(case['contexts']) != 5 or len({d['doc_id'] for d in case['contexts']}) != 5:
            raise ValueError('Cada consulta deve ter cinco documentos únicos.')
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = generator.manifest()
    signature = _digest({'bundle': bundle, 'manifest': manifest,
                         'config': asdict(generator.config)})
    status = {'status': 'incomplete', 'expected_responses': 20,
              'completed_responses': 0, 'experiment_signature': signature,
              'manifest': manifest}
    _write_json(output_dir / 'run_status.json', status)
    rows = []
    for case in cases:
        qid, query = case['query_id'], case['query']
        for condition in ('A_without_rag', 'B_with_rag'):
            with_rag = condition == 'B_with_rag'
            messages = (build_rag_messages(query, case['contexts'], generator.config.language)
                        if with_rag else build_no_rag_messages(query, generator.config.language))
            request_hash = _digest({'experiment': signature, 'messages': messages})
            path = output_dir / 'responses' / f'{qid}_{condition}.json'
            record = json.loads(path.read_text(encoding='utf-8')) if path.exists() else None
            if record is None or record.get('request_sha256') != request_hash:
                print(f'Gerando {qid}: {condition}', flush=True)
                generation = generator.generate(messages)
                record = {
                    'query_id': qid, 'query': query, 'condition': condition,
                    'external_context_provided': with_rag,
                    'contexts': case['contexts'] if with_rag else [],
                    'messages': messages, **generation,
                    'manifest': manifest, 'request_sha256': request_hash,
                    'experiment_signature': signature,
                    'created_at_utc': datetime.now(timezone.utc).isoformat(),
                }
                _write_json(path, record)
            rows.append(record)
            status['completed_responses'] = len(rows)
            _write_json(output_dir / 'run_status.json', status)
            print(f'Concluídas {len(rows)}/20; saída: {record["output_tokens"]} tokens.', flush=True)

    with (output_dir / 'answers.jsonl').open('w', encoding='utf-8') as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + '\n')
    columns = ['query_id', 'query', 'condition', 'external_context_provided',
               'answer', 'input_tokens', 'output_tokens', 'generation_seconds']
    with (output_dir / 'answers.csv').open('w', encoding='utf-8', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=columns, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)
    comparison = ['# Item 12 — Sem RAG × com RAG\n',
                  'Respostas experimentais; avaliação humana ainda não realizada.\n']
    for index, case in enumerate(cases):
        a, b = rows[index * 2:index * 2 + 2]
        comparison.extend([
            f'## {case["query_id"]} — {case["query"]}\n',
            '### A — Sem RAG\n\nNenhum contexto externo fornecido.\n\n' + a['answer'] + '\n',
            '### B — Com RAG\n\n' + b['answer'] + '\n',
            'Fontes enviadas: ' + ', '.join(d['doc_id'] for d in case['contexts']) + '\n',
        ])
    (output_dir / 'comparison.md').write_text('\n'.join(comparison), encoding='utf-8')
    export_standardized_csv(rows, output_dir / 'rag_experiment.csv')
    status['status'] = 'complete'
    _write_json(output_dir / 'run_status.json', status)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('prepare', 'generate'))
    parser.add_argument('--data-dir', default='nfcorpus/nfcorpus')
    parser.add_argument('--output-dir', default='Experiments/rag_experiment')
    parser.add_argument('--revision', default=RAGConfig().revision)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default=None)
    parser.add_argument('--threads', type=int, default=4)
    args = parser.parse_args()
    import torch
    torch.set_num_threads(args.threads)
    output_dir = Path(args.output_dir)
    if args.stage == 'prepare':
        prepare_contexts(args.data_dir, output_dir)
    else:
        bundle = json.loads((output_dir / 'contexts.json').read_text(encoding='utf-8'))
        config = RAGConfig(revision=args.revision)
        generator = LocalCausalGenerator(config, device=args.device)
        run_paired_experiment(bundle, generator, output_dir)


if __name__ == '__main__':
    main()
