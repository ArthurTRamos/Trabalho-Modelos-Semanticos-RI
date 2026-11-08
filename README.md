# Trabalho 2 — Modelos Semânticos em Recuperação da Informação

Pipeline de recuperação de informação sobre o **NFCorpus** (BEIR): busca lexical (BM25), tolerância a erros de digitação, recuperação semântica, fusão híbrida (RRF), reranking e geração com RAG, com avaliação quantitativa e humana.

## 1. Integrantes do grupo

| Nome completo | Nº USP |
|---|---|
| Arthur Trottmann Ramos | 14681052 |
| Maicon Chaves Marques | 14593530 |

## 2. Instruções para instalação e execução

### Instalação

1. Clone o repositório e entre na pasta:

   ```bash
   git clone <URL-DO-REPOSITORIO>
   cd Trabalho-Modelos-Semanticos-RI
   ```

2. Crie e ative um ambiente virtual com Python 3.13:

   ```bash
   python -m venv .venv
   source .venv/bin/activate        # Linux/macOS
   .venv\Scripts\activate           # Windows
   ```

3. Instale as dependências. Para GPU NVIDIA, instale antes o `torch` com a versão de CUDA adequada, conforme as instruções em pytorch.org.

   ```bash
   pip install nltk numpy pandas matplotlib scikit-learn gdown \
       "sentence-transformers==6.1.0" "transformers==5.20.1" \
       "accelerate==1.15.0" torch
   ```

4. Baixe o [NFCorpus (BEIR)](https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/nfcorpus.zip) e extraia o `.zip` na raiz do projeto, de modo que fique a pasta `nfcorpus/`.

### Execução

Execute com o **diretório de trabalho na raiz do projeto**, pois os caminhos em `ri_project/config.py` são relativos a ele (`nfcorpus/nfcorpus` para os dados e `Resultados/` para as saídas).

```bash
jupyter notebook Trabalho2.ipynb
```

Rode as células em ordem, de cima para baixo (veja a seção 10).

> **Atenção:** o projeto usa Sentence-BERT (rede neural), um cross-encoder e uma LLM, todos executados localmente. **É fortemente recomendado ter uma GPU NVIDIA com CUDA**; em CPU a execução pode levar horas.

## 3. Versão da linguagem

Python 3.13 (execução final: 3.13.16).

## 4. Bibliotecas e versões principais

| Biblioteca | Versão |
|---|---|
| torch | 2.14.1 (CUDA 13.0) |
| transformers | 5.20.1 |
| sentence-transformers | 6.1.0 |
| accelerate | 1.15.0 |
| numpy | 2.4.6 |
| pandas | 2.2.3 |
| nltk | stopwords e stemmer de Porter |
| scikit-learn, matplotlib, gdown | versão atual no momento da instalação |

As versões da execução final estão registradas em `Resultados/colab_execution/`.

## 5. Modelo lexical escolhido

**BM25**, implementado do zero sobre um índice invertido próprio (`índice[termo][doc_id] = tf`), com IDF suavizado `ln(1 + (N − df + 0,5) / (df + 0,5))`, k1 = 1,2 e b = 0,75, sem ajuste.

Pré-processamento: minúsculas, tokenização com `\b\w+\b`, remoção de stopwords em inglês e stemming de Porter, aplicado igualmente a documentos e consultas.

## 6. Modelo de embeddings

`sentence-transformers/all-MiniLM-L6-v2` (revisão `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`): bi-encoder com 384 dimensões, embeddings normalizados em L2, similaridade por cosseno e comprimento máximo de 256 tokens.

## 7. Modelo de reranking

`cross-encoder/ms-marco-MiniLM-L6-v2` (revisão `233902d25c440f23af6f7d6e94d2946bac0bee0a`), aplicado ao Top-50 do ranking híbrido (RRF, k = 10), com comprimento máximo de 512 tokens.

## 8. LLM utilizada

`Qwen/Qwen2.5-1.5B-Instruct` (revisão `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`), executada localmente, sem API externa. Temperatura 0 (decodificação gulosa), semente 42, máximo de 512 tokens gerados, 5 documentos integrais no contexto, instruções e respostas em inglês.

## 9. Hardware empregado

Máquina local com GPU **NVIDIA GeForce RTX 4050 Laptop (6 GB)** e CPU **Intel Core i5-13420H**. O modelo de embeddings, o reranker e a LLM rodam em CUDA (a LLM em BF16). Uma GPU com 6 GB de VRAM ou mais é suficiente. O BM25, o índice e a correção ortográfica rodam apenas em CPU.

## 10. Ordem recomendada de execução

Todas as etapas estão em `Trabalho2.ipynb`, que deve ser executado de cima para baixo:

| # | Etapa (seção do notebook) | Saídas principais em `Resultados/` |
|---|---|---|
| 1 | Instalação de dependências | — |
| 2 | Pré-processamento e leitura do NFCorpus | — |
| 3 | Índice invertido | — |
| 4 | Busca lexical (BM25) | — |
| 5 | Tolerância a erros: ajuste em `dev` e experimento em `test` | `typo_dev_tuning.csv`, `typo_experiment.csv` |
| 6 | Recuperação semântica (embeddings em cache) | `semantic/all_minilm_l6_v2_*` |
| 7 | Busca híbrida: ajuste de k do RRF em `dev` | `rrf_dev_tuning.csv`, `rrf_dev_tuning_per_query.csv` |
| 8 | Reranking do Top-50 (validação em `dev`) | `reranking_dev/`, `reranker_manifest.json` |
| 9 | Avaliação lexical, semântica, híbrida e com reranking em `test` | `results_lexical.csv`, `results_semantic.csv`, `results_hybrid.csv`, `results_reranking.csv`, `metrics_per_query.csv`, `metrics_summary.csv` |
| 10 | Casos de comparação lexical × semântico | `comparison_cases_summary.csv`, `comparison_cases_top5.csv` |
| 11 | Análise por consulta do reranking | `reranking_analysis_*.csv` |
| 12 | Pipeline RAG (exemplo em `dev`) | `rag/` |
| 13 | Experimento sem RAG × com RAG (20 respostas) | `rag_experiment/`, `rag_experiment.csv` |
| 14 | Avaliação humana do RAG e análise de falhas | `rag_experiment/evaluation/`, `rag_experiment.csv` (com as notas) |

As etapas 1 a 11 não dependem da LLM; as etapas 12 e 13 carregam o Qwen2.5-1.5B-Instruct na GPU.

---

## Estrutura do repositório

```
.
├── Trabalho2.ipynb        # Notebook principal
├── ri_project/            # Implementação
│   ├── config.py          # Caminhos e parâmetros
│   ├── data.py            # Leitura de corpus, consultas e qrels
│   ├── preprocessing.py   # Tokenização, stopwords e stemming
│   ├── indexing.py        # Índice invertido
│   ├── bm25.py            # Busca lexical
│   ├── typo.py            # k-gramas, Jaccard, edição e experimento de erros
│   ├── semantic.py        # Bi-encoder e embeddings
│   ├── hybrid.py          # Fusão RRF
│   ├── reranking.py       # Cross-encoder
│   ├── evaluation.py      # Métricas, comparações e análise do reranking
│   ├── rag.py             # Pipeline RAG e gerador local
│   └── rag_experiment.py  # Experimento sem RAG × com RAG
├── Resultados/            # Arquivos de resultados e artefatos
└── nfcorpus/nfcorpus/     # Dataset (não versionado)
```

## Arquivos de resultados padronizados

Todos em `Resultados/`, independentes das saídas do notebook:

| Arquivo | Conteúdo |
|---|---|
| `results_lexical.csv`, `results_semantic.csv`, `results_hybrid.csv`, `results_reranking.csv` | Top-100 de cada uma das 323 consultas de teste: `query_id`, `doc_id`, `rank`, `score`, `relevance` |
| `metrics_per_query.csv` | `query_id`, `model`, `P10`, `R10`, `AP10`, `NDCG10` |
| `metrics_summary.csv` | Médias por modelo (a média de `AP10` é o MAP@10) |
| `typo_experiment.csv` | Experimento de erros em 10 consultas, com NDCG@10 original, com erro e corrigido |
| `rag_experiment.csv` | Condições A (sem RAG) e B (com RAG): contexto, prompt, resposta, quatro notas e justificativas |

## Resultados resumidos (teste, 323 consultas)

| Sistema | P@10 | R@10 | MAP@10 | NDCG@10 |
|---|---:|---:|---:|---:|
| Lexical (BM25) | 0,2390 | 0,1532 | 0,1218 | 0,3278 |
| Semântico (all-MiniLM-L6-v2) | 0,2433 | 0,1550 | 0,1108 | 0,3167 |
| Híbrido (RRF, k = 10) | 0,2567 | 0,1698 | 0,1326 | 0,3505 |
| Híbrido + reranking | 0,2511 | 0,1666 | 0,1388 | 0,3564 |
