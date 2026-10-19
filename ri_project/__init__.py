"""Módulos do projeto de recuperação da informação."""

from .config import DATA_DIR, EXPERIMENTS_DIR
from .data import load_dataset, read_jsonl
from .indexing import build_inverted_index
from .preprocessing import preprocess, preprocess_tokens, tokenize

__all__ = [
    'DATA_DIR', 'EXPERIMENTS_DIR', 'load_dataset', 'read_jsonl',
    'build_inverted_index', 'preprocess', 'preprocess_tokens', 'tokenize',
]
