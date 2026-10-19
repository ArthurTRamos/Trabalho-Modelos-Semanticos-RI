"""Leitura dos arquivos JSONL e qrels do NF-Corpus."""
import json
from pathlib import Path

import pandas as pd

from .config import DATA_DIR


def read_jsonl(path):
    """Lê JSON Lines e indexa os registros pelo campo ``_id``."""
    path = Path(path)
    records = {}
    with path.open(encoding='utf-8') as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            record_id = record.get('_id')
            if record_id is None:
                raise ValueError(f'Registro sem _id em {path}, linha {line_number}')
            records[record_id] = record
    return records


def load_dataset(data_dir=DATA_DIR):
    """Carrega documentos, consultas e qrels train/dev/test."""
    data_dir = Path(data_dir)
    documents = read_jsonl(data_dir / 'corpus.jsonl')
    queries = read_jsonl(data_dir / 'queries.jsonl')
    qrels = {
        split: pd.read_csv(data_dir / 'qrels' / f'{split}.tsv', sep='\t')
        for split in ('train', 'dev', 'test')
    }
    return {
        'documents': documents,
        'queries': queries,
        'qrels': qrels,
        'qrels_all': pd.concat(qrels.values(), ignore_index=True),
    }
