"""Caminhos e configurações compartilhadas do projeto."""
from pathlib import Path

PROJECT_ROOT = Path.cwd()
DATA_DIR = PROJECT_ROOT / 'nfcorpus' / 'nfcorpus'
EXPERIMENTS_DIR = PROJECT_ROOT / 'Experiments'
EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)

BM25_K1 = 1.2
BM25_B = 0.75
SEMANTIC_MODEL_NAME = 'sentence-transformers/all-MiniLM-L6-v2'
SEMANTIC_MODEL_REVISION = '1110a243fdf4706b3f48f1d95db1a4f5529b4d41'
SEMANTIC_BATCH_SIZE = 32
SEMANTIC_NORMALIZE_EMBEDDINGS = True
EVALUATION_SPLIT = 'test'
EVALUATION_K = 10
EVALUATION_RANKING_DEPTH = 100
