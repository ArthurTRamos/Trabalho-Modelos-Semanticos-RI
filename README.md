# Trabalho-Modelos-Semanticos-RI

**Grupo**:

- Arthur Trottmann Ramos (14681052)
- Maicon Chaves Marques (14593530)

Link de acesso ao relatório: https://docs.google.com/document/d/1tfYRqScAYURl9fCsafOWWNqWjfrmm_A7AEU_h4iyl6M/edit?usp=sharing

## Descrição

Projeto do Trabalho Prático 2 de Recuperação da Informação. O notebook coordena os experimentos e o pacote `ri_project/` contém as implementações reutilizáveis de leitura, pré-processamento, indexação invertida, BM25, tolerância a erros, recuperação semântica e avaliação.

## Como Executar

1. Coloque o NF-Corpus no caminho `nfcorpus/nfcorpus/` (com `corpus.jsonl`, `queries.jsonl` e `qrels/`).
2. Instale as dependências em um ambiente Python com suporte a Jupyter:

   ```bash
   pip install jupyter nltk numpy pandas matplotlib sentence-transformers scikit-learn
   ```

3. Na raiz do projeto, inicie Jupyter e abra `Trabalho2.ipynb`.
4. Execute as células em ordem. A primeira execução pode baixar stopwords do NLTK e o modelo Sentence-BERT. O notebook cria `Experiments/` para embeddings e arquivos gerados.

`Experiments/` é ignorada pelo Git; execute novamente o notebook para regenerar os resultados locais.

## Organização do código

- `ri_project/config.py`: caminhos e parâmetros compartilhados.
- `ri_project/data.py`: leitura dos documentos, consultas e qrels.
- `ri_project/preprocessing.py`: tokenização, stopwords e stemming.
- `ri_project/indexing.py`: índice invertido e estatísticas.
- `ri_project/bm25.py`: busca lexical BM25 e medição de latência.
- `ri_project/typo.py`: k-gramas, Jaccard, distância de edição, correção e ajuste em Dev.
- `ri_project/semantic.py`: embeddings, cache validado e busca Sentence-BERT.
- `ri_project/evaluation.py`: métricas, rankings e comparação por consulta.
- `Trabalho2.ipynb`: fluxo de execução e visualização dos resultados.

## Linguagem e Bibliotecas

Python; NLTK, NumPy, pandas, Matplotlib, scikit-learn e sentence-transformers.

## Modelos

BM25 para recuperação lexical e `sentence-transformers/all-MiniLM-L6-v2` para recuperação semântica. Os embeddings documentais são pré-calculados e armazenados localmente em `Experiments/semantic/`.

## Hardware e Ordem de Execução

A busca semântica seleciona CUDA quando disponível e usa CPU como alternativa. Execute as células do notebook de cima para baixo: leitura e pré-processamento, índice/BM25, ajuste e avaliação de tolerância a erros, embeddings semânticos, métricas e comparação lexical-semântica.
