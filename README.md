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
   pip install jupyter nltk numpy pandas matplotlib "sentence-transformers==6.1.0" scikit-learn "transformers==5.20.1" torch "accelerate==1.15.0"
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
- `ri_project/rag_evaluation.py`: métricas de contexto, fichas mascaradas, validação das notas humanas, médias e concordância.
- `ri_project/rag_analysis.py`: diagnóstico por etapa, sugestões de casos e relatório com evidências conferidas no corpus.
- `RAG_Colab.ipynb`: geração das vinte respostas em GPU remota, com código e contextos incorporados.
- Rankings híbridos e ajuste ficam em `Experiments/results_hybrid.csv` e `Experiments/rrf_dev_tuning*.csv`; reranking em `Experiments/results_reranking.csv`.
- `ri_project/evaluation.py`: métricas, rankings e comparação por consulta.
- `Trabalho2.ipynb`: fluxo de execução e visualização dos resultados.

## Linguagem e Bibliotecas

Python 3.10 ou superior (respeite também os requisitos das versões de bibliotecas instaladas); NLTK, NumPy, pandas, Matplotlib, scikit-learn, sentence-transformers, PyTorch, Accelerate e Transformers. A execução local de 2026-10-26 usa Python 3.14.4, PyTorch 2.14.1+cu130, Transformers 5.20.1, sentence-transformers 6.1.0 e Accelerate 1.15.0. O manifesto do RAG registra Python, PyTorch e Transformers; `Experiments/execution_2026-10-26/environment.json` registra as demais versões e a GPU. O notebook Colab instala suas próprias versões e registra o ambiente remoto separadamente.

## Modelos

BM25 para recuperação lexical, `sentence-transformers/all-MiniLM-L6-v2` para recuperação semântica e fusão híbrida por RRF, com `k` ajustado em `dev` entre 10, 60 e 100, e cross-encoder `cross-encoder/ms-marco-MiniLM-L6-v2` para reranking do Top-50. Os embeddings documentais são pré-calculados e armazenados localmente em `Experiments/semantic/`.

## Hardware e Ordem de Execução

A busca semântica e a LLM selecionam CUDA quando disponível e usam CPU como alternativa. Execute as células do notebook de cima para baixo: leitura e pré-processamento, índice/BM25, ajuste e avaliação de tolerância a erros, embeddings semânticos, ajuste RRF em `dev`, avaliação do cross-encoder em `dev`, avaliação comparativa em `test`, comparação dos métodos, geração com RAG, métricas do contexto, consolidação das notas humanas e análise de falhas. O manifesto da LLM informa CPU/GPU, dispositivo e dtype efetivamente usados.

A execução local de 2026-10-26 utiliza a NVIDIA GeForce RTX 4050 Laptop GPU (6 GB), com a LLM em BF16. O notebook executado e o log ficam em `Experiments/execution_2026-10-26/`. Os resultados híbridos e de reranking foram recalculados após corrigir a complementação de listas já completas: a fusão agora utiliza exatamente os Top-100 de cada método. Os números e as análises anteriores à correção foram preservados nessa pasta como histórico e não devem ser usados como resultado final.

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

Cada resposta é salva imediatamente em `responses/`. `run_status.json` indica `incomplete` ou `complete` e a contagem concluída; somente `complete` com 20 respostas caracteriza a execução finalizada. Ao concluir, são exportados `answers.jsonl`, `answers.csv`, `comparison.md` e `rag_experiment.csv`. Este último segue o item 15: condições A/B, IDs de contexto, prompt, resposta e campos das quatro notas/justificativas. Notas humanas pendentes ficam em branco; rastreabilidade de A fica `N/A`. Reexecutar `generate` reutiliza checkpoints somente se configuração, contexto, prompt e manifesto forem idênticos. Alterações de hardware/versões/contexto invalidam a reutilização. Notas já preenchidas são preservadas apenas quando consulta, prompt, resposta e documentos de contexto permanecem iguais. A avaliação e análise dos itens 13–14 são descritas abaixo; as notas exigem preenchimento humano.

Após a geração, confira o pareamento e a integridade das fontes e exportações:

```bash
python scripts/verify_rag_experiment.py
```

## Avaliação e análise do RAG — itens 13–14

As seções finais do notebook executam esta fase sobre os resultados salvos, sem carregar modelos ou gerar novas respostas. Também é possível executar por CLI:

```bash
python -m ri_project.rag_evaluation prepare
```

Os artefatos ficam em `Experiments/rag_experiment/evaluation/`. `context_metrics_per_query.csv` registra Precision@5 (`P5`), Recall@5 (`R5`) e Hit@5 (`Hit5`) dos cinco documentos efetivamente enviados; `context_metrics_summary.csv` guarda médias macro das dez consultas. Recall usa todos os documentos de grau positivo nos qrels. Na execução de 2026-10-26: P@5 0,46; R@5 0,1075; Hit@5 0,90. Esses valores se referem às dez consultas do RAG, não à avaliação Top-10 das 323 consultas.

Para a avaliação humana, cada integrante deve ler apenas `human_review.md` e preencher sua ficha `ratings_reviewer_1.csv` ou `ratings_reviewer_2.csv`, mantendo IDs e justificando as notas 0–2. O segundo avaliador é opcional. A ordem das vinte respostas é embaralhada com seed 42 e os rótulos de condição ficam em `blind_key.json`, que não deve ser consultado antes da avaliação. Estilo e citações podem sugerir a condição, portanto o mascaramento não garante cegamento completo. Os mesmos Top-5 são apresentados como referência para avaliar a fidelidade das duas respostas de cada pergunta. Rastreabilidade é avaliada provisoriamente na ficha mascarada e, ao revelar a condição A, substituída por N/A e excluída da média. Não há atribuição automática de notas de qualidade.

Após preencher uma ficha:

```bash
python -m ri_project.rag_evaluation aggregate --ratings Experiments/rag_experiment/evaluation/ratings_reviewer_1.csv
```

Com duas fichas completas, informe ambas e escolha explicitamente as notas finais com `--final-reviewer avaliador_1` (ou o ID adotado pelo grupo). As médias são reportadas separadamente por avaliador, condição e critério; a concordância exata e o kappa de Cohen com pesos quadráticos ficam em `human_agreement.csv`. Kappa sem variabilidade marginal permanece indefinido. Médias de rascunhos têm contagem e estado de incompletude; campos vazios não viram zero. Ao concluir a avaliação escolhida, `rag_experiment.csv` recebe notas inteiras e justificativas, e são gerados `human_evaluation_report.md` e `quality_comparison_per_query.csv`. Duas notas diferentes não são convertidas automaticamente em uma nota fracionária.

O item 14 possui dossiers das dez perguntas em `failure_analysis_dossiers.md`, contagens por etapa em `stage_metrics_per_query.csv` e posições/textos em `stage_document_evidence.csv`, incluindo relevantes ausentes dos rankings. O diagnóstico confere a fusão com os baselines e o ranking final com o contexto real antes de atribuir posições às etapas. A preparação preserva fichas existentes e rejeita sua reutilização para outras respostas/contextos; os manifestos registram hashes dos dados e diagnósticos.

Após consolidar as notas, `failure_case_selection.json` sugere três consultas distintas pela diferença B−A na média de Fidelidade, Completude e Utilidade, com pesos iguais. A maior diferença positiva sugere melhora; a menor não positiva sugere ausência de ajuda; o menor valor absoluto de até 1/3, com no máximo um ponto de mudança total entre os três critérios, sugere pouca diferença. Essa convenção auxilia a seleção e não é exigência do enunciado. Categorias não observadas ficam explícitas. O grupo confirma ou altera os casos em `case_reviews.json`, justificando a escolha, comparando os textos e indicando se a informação necessária chegou ao contexto e em qual etapa ocorreu o problema. Cada evidência contém doc_id e trecho literal do corpus. Trechos enviados e não enviados são identificados no relatório.

```bash
python -m ri_project.rag_evaluation analyze
```

`failure_analysis.md` somente é publicado após notas e revisões humanas completas. Campos pendentes produzem um arquivo de estado, sem inventar casos ou conclusões. Um rascunho qualitativo com apoio de IA para os resultados atuais está em `failure_analysis_preliminary.md`; ele não contém notas humanas, suas categorias são provisórias e não deve ser mostrado aos avaliadores antes da etapa mascarada. Todos os resultados continuam em `Experiments/`, ignorada pelo Git.

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
