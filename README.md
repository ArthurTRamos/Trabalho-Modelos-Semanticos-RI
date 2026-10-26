# Trabalho-Modelos-Semanticos-RI

**Grupo**:

- Arthur Trottmann Ramos (14681052)
- Maicon Chaves Marques (14593530)

Link de acesso ao relatório: https://docs.google.com/document/d/1tfYRqScAYURl9fCsafOWWNqWjfrmm_A7AEU_h4iyl6M/edit?usp=sharing

## Descrição

Projeto do Trabalho Prático 2 de Recuperação da Informação. O notebook coordena os experimentos e o pacote `ri_project/` contém as implementações reutilizáveis de leitura, pré-processamento, indexação invertida, BM25, tolerância a erros, recuperação semântica, fusão RRF, reranking, avaliação e geração com RAG.

## Como Executar

1. Coloque o NF-Corpus no caminho `nfcorpus/nfcorpus/` (com `corpus.jsonl`, `queries.jsonl` e `qrels/`).
2. Instale as dependências em um ambiente Python com suporte a Jupyter:

   ```bash
   pip install jupyter nltk numpy pandas matplotlib sentence-transformers scikit-learn "transformers>=4.57,<5" torch accelerate
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
- `ri_project/rag.py`: seleção de documentos integrais, prompt com fontes, geração local e exportação auditável.
- `ri_project/rag_experiment.py`: sorteio oficial e experimento pareado sem/com RAG, com checkpoints.
- `RAG_Colab.ipynb`: geração das vinte respostas em GPU remota, com código e contextos incorporados.
- Rankings híbridos e ajuste ficam em `Experiments/results_hybrid.csv` e `Experiments/rrf_dev_tuning*.csv`; reranking em `Experiments/results_reranking.csv`.
- `ri_project/evaluation.py`: métricas, rankings e comparação por consulta.
- `Trabalho2.ipynb`: fluxo de execução e visualização dos resultados.

## Linguagem e Bibliotecas

Python 3.10 ou superior (respeite também os requisitos das versões de bibliotecas instaladas); NLTK, NumPy, pandas, Matplotlib, scikit-learn, sentence-transformers, PyTorch, Accelerate e Transformers 4.x (>=4.57). O manifesto do RAG registra as versões efetivamente utilizadas de Python, PyTorch e Transformers. A execução local de 2026-10-26 usa Python 3.14.4, PyTorch 2.14.1+cpu, Transformers 4.57.6 e sentence-transformers 5.7.0; a lista completa está em `Experiments/rag_experiment/environment_cpu.txt`.

## Modelos

BM25 para recuperação lexical, `sentence-transformers/all-MiniLM-L6-v2` para recuperação semântica e fusão híbrida por RRF, com `k` ajustado em `dev` entre 10, 60 e 100, e cross-encoder `cross-encoder/ms-marco-MiniLM-L6-v2` para reranking do Top-50. Os embeddings documentais são pré-calculados e armazenados localmente em `Experiments/semantic/`.

## Hardware e Ordem de Execução

A busca semântica e a LLM selecionam CUDA quando disponível e usam CPU como alternativa. Execute as células do notebook de cima para baixo: leitura e pré-processamento, índice/BM25, ajuste e avaliação de tolerância a erros, embeddings semânticos, ajuste RRF em `dev`, avaliação do cross-encoder em `dev`, avaliação comparativa em `test`, comparação dos métodos e geração com RAG. O manifesto da LLM informa CPU/GPU, dispositivo e dtype efetivamente usados.

## RAG — item 11

A seção de RAG do notebook inicializa e demonstra `RAGPipeline` em uma consulta de `dev`. O pipeline usa o recuperador híbrido já configurado, aplica reranking ao Top-50 e envia os cinco primeiros documentos, com `doc_id`, título e texto integral, à LLM. O contexto não usa qrels para escolher ou filtrar documentos. A instrução exige respostas apoiadas nas fontes, citações `[doc_id]` e declaração de evidência insuficiente.

A escolha inicial é [Qwen/Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct), executado localmente, em inglês, com temperatura configurada 0 (greedy, `do_sample=False`), seed 42 e limite de 512 novos tokens. A configuração é uma decisão de implementação, sem comparação de alternativas em `test`. Não há cobrança de API; custos de energia/hardware não foram estimados. A primeira execução precisa obter os pesos do Hugging Face; a execução em CPU pode ser demorada.

Cada chamada pode salvar um JSON em `Experiments/rag/` com resposta, documentos integrais na ordem enviada, mensagens, prompt formatado, tokens, latências, configuração de recuperação e manifesto da LLM. A revisão da LLM foi fixada em `989aa7980e4cf806f80c7fef2b1adb7bc71aa306` e é confirmada em `resolved_model_revision`. O prompt completo é medido com o tokenizador da LLM, reservando espaço para a saída. Caso não caiba na janela, a execução falha explicitamente, sem truncar documentos ou substituir o Top-5.

## Experimento sem RAG × com RAG — item 12

O sorteio é conferido contra a lista obrigatória: PLAIN-68, PLAIN-2450, PLAIN-153, PLAIN-2850, PLAIN-2750, PLAIN-270, PLAIN-2470, PLAIN-2430, PLAIN-238 e PLAIN-3342. São geradas duas respostas para cada pergunta usando os mesmos pesos, idioma e parâmetros. A condição A recebe a pergunta e a instrução de idioma; a B acrescenta os cinco documentos integrais após reranking e instruções de uso de evidências. Cada geração começa uma conversa nova, sem histórico da outra condição.

Para executar apenas as etapas necessárias, sem repetir os experimentos anteriores do notebook, use dois processos sequenciais; isso libera os modelos de recuperação antes de carregar a LLM:

```bash
python -m ri_project.rag_experiment prepare
python -m ri_project.rag_experiment generate
```

`prepare` carrega o NF-Corpus, gera/reutiliza embeddings validados, ajusta `k` somente em `dev` e salva o Top-5 e o Top-100 reranqueado das dez perguntas em `Experiments/rag_experiment/contexts.json`, junto de hashes dos dados e parâmetros. `generate` usa CUDA se disponível, CPU caso contrário. Para CUDA, a LLM usa BF16 quando suportado, FP16 nos demais casos (incluindo T4); em CPU usa FP32. O hardware e a precisão ficam registrados. Para manter os caches dentro do repositório, configure `HF_HOME`, `NLTK_DATA` e `MPLCONFIGDIR` para subpastas de `Experiments/cache/` antes de executar.

Cada resposta é salva imediatamente em `responses/`. `run_status.json` indica `incomplete` ou `complete` e a contagem concluída; somente `complete` com 20 respostas caracteriza a execução finalizada. Ao concluir, são exportados `answers.jsonl`, `answers.csv`, `comparison.md` e `rag_experiment.csv`. Este último segue o item 15: condições A/B, IDs de contexto, prompt, resposta e campos das quatro notas/justificativas. Notas humanas pendentes ficam em branco; rastreabilidade de A fica `N/A`. Reexecutar `generate` reutiliza checkpoints somente se configuração, contexto, prompt e manifesto forem idênticos. Alterações de hardware/versões/contexto invalidam a reutilização. As respostas não recebem avaliação humana automática: os itens 13–14 continuam pendentes.

Após a geração, confira o pareamento e a integridade das fontes e exportações:

```bash
python scripts/verify_rag_experiment.py
```

## Execução na extensão Colab

Abra `RAG_Colab.ipynb`, selecione **Select Kernel → Colab** e um runtime com GPU e execute todas as células. A extensão e sua autenticação são operadas pelo VS Code; não são controladas pelo terminal local desta sessão. O notebook verifica CUDA antes de gerar, inclui os contextos já recuperados e um snapshot do código validado por SHA-256 e não depende de arquivos locais acessíveis pelo runtime remoto. O empacotamento usa apenas as fontes recuperadas, sem respostas ou pesos de modelos.

As vinte respostas da execução GPU ficam em `Experiments/rag_experiment_gpu/` no runtime e podem ser baixadas no ZIP produzido na última célula ou pelo explorador Colab, opção **Download**. Salve o ZIP antes de encerrar o runtime. Uma execução GPU é um experimento separado da execução CPU; não combine respostas obtidas em precisões/dispositivos diferentes na mesma comparação pareada.

Depois de alterar o código ou regenerar os contextos, atualize o snapshot:

```bash
python scripts/build_colab_rag.py
```

Para verificar os contratos de integração offline (componentes simulados, sem inferência real):

```bash
python3 -m unittest discover -s tests -v
```
