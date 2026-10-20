# Trabalho-Modelos-Semanticos-RI

**Grupo**:

- Arthur Trottmann Ramos (14681052)
- Maicon Chaves Marques (14593530)

Link de acesso ao relatório: https://docs.google.com/document/d/1tfYRqScAYURl9fCsafOWWNqWjfrmm_A7AEU_h4iyl6M/edit?usp=sharing

## Descrição

Projeto do Trabalho Prático 2 de Recuperação da Informação. O notebook coordena os experimentos e o pacote `ri_project/` contém as implementações reutilizáveis de leitura, pré-processamento, indexação invertida, BM25, tolerância a erros, recuperação semântica, fusão RRF, reranking e avaliação.

## Como Executar

1. Coloque o NF-Corpus no caminho `nfcorpus/nfcorpus/` (com `corpus.jsonl`, `queries.jsonl` e `qrels/`).
2. Instale as dependências em um ambiente Python com suporte a Jupyter:

   ```bash
   pip install jupyter nltk numpy pandas matplotlib sentence-transformers scikit-learn
   ```

3. Na raiz do projeto, inicie Jupyter e abra `Trabalho2.ipynb`.
4. Execute as células em ordem. A primeira execução pode baixar stopwords do NLTK, os modelos Sentence-BERT e cross-encoder. O notebook cria `Experiments/` para embeddings e arquivos gerados.

`Experiments/` é ignorada pelo Git; execute novamente o notebook para regenerar os resultados locais.

## Organização do código

- `ri_project/config.py`: caminhos e parâmetros compartilhados.
- `ri_project/data.py`: leitura dos documentos, consultas e qrels.
- `ri_project/preprocessing.py`: tokenização, stopwords e stemming.
- `ri_project/indexing.py`: índice invertido e estatísticas.
- `ri_project/bm25.py`: busca lexical BM25 e medição de latência.
- `ri_project/typo.py`: k-gramas, Jaccard, distância de edição, correção e ajuste em Dev.
- `ri_project/semantic.py`: embeddings, cache validado e busca Sentence-BERT.
- `ri_project/hybrid.py`: fusão de rankings lexical e semântico com Reciprocal Rank Fusion (RRF).
- `ri_project/reranking.py`: reranking do Top-50 híbrido com cross-encoder e preservação da cauda.
- Rankings híbridos e ajuste ficam em `Experiments/results_hybrid.csv` e `Experiments/rrf_dev_tuning*.csv`; reranking em `Experiments/results_reranking.csv`.
- `ri_project/evaluation.py`: métricas, rankings e comparação por consulta.
- `Trabalho2.ipynb`: fluxo de execução e visualização dos resultados.

## Linguagem e Bibliotecas

Python; NLTK, NumPy, pandas, Matplotlib, scikit-learn e sentence-transformers.

## Modelos

BM25 para recuperação lexical, `sentence-transformers/all-MiniLM-L6-v2` para recuperação semântica e fusão híbrida por RRF, com `k` ajustado em `dev` entre 10, 60 e 100, e cross-encoder `cross-encoder/ms-marco-MiniLM-L6-v2` para reranking do Top-50. Os embeddings documentais são pré-calculados e armazenados localmente em `Experiments/semantic/`.

## Hardware e Ordem de Execução

A busca semântica seleciona CUDA quando disponível e usa CPU como alternativa. Execute as células do notebook de cima para baixo: leitura e pré-processamento, índice/BM25, ajuste e avaliação de tolerância a erros, embeddings semânticos, ajuste RRF em `dev`, avaliação do cross-encoder em `dev`, avaliação comparativa em `test` e comparação dos métodos.
