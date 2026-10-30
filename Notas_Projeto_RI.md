# Trabalho Prático 2 — Recuperação da Informação

Notas de implementação e decisões para apoiar a redação do relatório.

## Identificação do projeto

| Campo | Valor |
|---|---|
| Projeto | `Trabalho-Modelos-Semanticos-RI` |
| Notebook | `Trabalho2.ipynb` |
| Dataset | `nfcorpus/nfcorpus/` |
| Documento de requisitos | `Trabalho_Pratico_2_RI_2026.md` (versão Markdown consultada) |

## Estado geral

**Execução atual — 2026-10-30:** Os artefatos experimentais anteriores foram excluídos sem backup por solicitação do usuário. Os registros abaixo datados anteriormente são históricos; referências a arquivos antigos não indicam que esses arquivos continuam disponíveis. Os resultados atuais e o escopo da limpeza estão na seção final “Limpeza e reexecução completa — 2026-10-30”. O antigo notebook Colab e seu gerador foram removidos nessa limpeza. Posteriormente foi criado `Trabalho2_Colab.ipynb`, com o fluxo completo, conforme registro ao final.

Estão implementados pré-processamento, índice invertido, BM25, tolerância a erros com ajuste em dev, recuperação semântica, comparação lexical × semântica, RRF ajustado em dev, reranking Top-50, análise por consulta e geração com/sem RAG nas dez perguntas oficiais (itens 1–12). O fluxo foi reexecutado integralmente na GPU local em 2026-10-30, após a limpeza, e as vinte respostas novas passaram pela verificação de pareamento e integridade das fontes. As métricas de contexto do item 13 foram calculadas. O fluxo de avaliação humana e análise de falhas dos itens 13–14 está implementado, com fichas, consolidação, diagnóstico e exportação final condicionada ao preenchimento humano. As notas reais e a confirmação dos três casos permanecem pendentes (Etapas 13–14).

**Correção de resultados em 2026-10-26:** A complementação de uma lista já completa continuava acrescentando documentos, levando a fusão RRF além dos Top-100 previstos no item 8. O limite foi corrigido e o ajuste em dev, a avaliação em test, o reranking e a análise por consulta foram refeitos. Os resultados atuais nas Etapas 7, 9, 10 e 12 substituem os valores híbridos/reranqueados anteriores. Esses artefatos antigos foram excluídos na limpeza de 2026-10-30; este registro da correção é histórico.

## Etapas de implementação

Cada etapa segue o mesmo registro: **objetivo e estado**, **implementação**, **decisões e justificativas**, **verificação e resultados**, **pendências**. Os campos de resultados devem refletir apenas execuções realizadas; se ainda não houve execução, registrar isso explicitamente.

### Etapa 1 — Leitura do dataset

**Estado:** Implementada e executada no experimento registrado em 2026-10-17.

**Objetivo:** Carregar corpus, consultas e julgamentos de relevância (qrels) do NF-Corpus.

**Implementação:**

- A pasta de dados é `nfcorpus/nfcorpus`, relativa à raiz do projeto, evitando caminhos absolutos específicos da máquina.
- `read_jsonl(path)` lê linha a linha em UTF-8, interpreta cada linha como JSON e devolve um dicionário indexado por `_id`.
- Linhas vazias são ignoradas. Registro sem `_id` produz erro explícito com caminho e número da linha.
- `corpus.jsonl` é carregado em `documents`; `queries.jsonl`, em `queries`.
- `qrels/train.tsv`, `qrels/dev.tsv` e `qrels/test.tsv` são carregados com pandas, mantidos em `qrels` por partição e concatenados em `qrels_all`. As colunas originais são preservadas.

**Decisões e justificativas:** Manter as partições separadas permite escolher explicitamente os dados de desenvolvimento e teste; `qrels_all` facilita inspeção geral sem perder essa separação.

**Verificação e resultados:** A execução usada para o experimento em 2026-10-18 carregou 3.633 documentos, 3.237 consultas e 134.294 linhas de qrels somadas das três partições. As colunas carregadas são `query-id`, `corpus-id` e `score`.

**Pendências:** Nenhuma pendência específica de leitura identificada nesta execução.

### Etapa 2 — Pré-processamento lexical

**Estado:** Implementada e executada no experimento registrado em 2026-10-17.

**Objetivo:** Aplicar a mesma normalização ao texto dos documentos e das consultas.

**Implementação:** A função `preprocess(text)` converte o conteúdo para texto e minúsculas, tokeniza com `\\b\\w+\\b`, remove stopwords em inglês e aplica stemming Porter.

**Decisões e justificativas:** A normalização comum reduz divergências de representação entre consultas e documentos. A remoção de stopwords reduz termos frequentes com baixo poder discriminativo; stemming aproxima variantes morfológicas. O conteúdo do NF-Corpus é majoritariamente em inglês.

**Dependências e limitações:** As stopwords vêm do corpus `stopwords` do NLTK. O notebook tenta carregá-las localmente e só solicita download se não estiverem disponíveis; as dependências do notebook são instaladas com `%pip install`.

**Verificação e resultados:** O pré-processamento foi executado sobre o corpus. Stopwords estavam disponíveis localmente. A chamada de download foi tornada condicional para evitar tentativa de rede quando o recurso já existe.

**Pendências:** Nenhuma pendência específica identificada nesta execução.

### Etapa 3 — Construção do índice invertido

**Estado:** Implementada e executada no experimento registrado em 2026-10-17.

**Objetivo:** Construir postings e estatísticas necessárias à busca lexical.

**Implementação:**

- O texto de cada documento é a concatenação de `title` e `text`, usando string vazia quando um campo estiver ausente.
- `inverted_index[termo][doc_id]` armazena a frequência do termo no documento (TF). O número de postings do termo corresponde à sua frequência documental (`document_frequency(term)`).
- São mantidas `document_lengths[doc_id]`, `num_documents`, `avg_document_length`, `num_postings` e `index_build_time_seconds`.
- O comprimento é contado após remoção de stopwords e stemming. `num_postings` conta pares termo-documento, não ocorrências individuais além do TF.
- O tempo medido cobre a criação das posting lists e os comprimentos dos documentos; não inclui a leitura dos arquivos nem o cálculo das estatísticas finais.
- Ao final, o `defaultdict` é convertido para `dict` comum.

**Decisões e justificativas:** Comprimento e postings usam a mesma representação processada que será consultada pelo BM25. Guardar TF nas posting lists permite calcular pontuações sem percorrer todo o corpus para cada consulta. Converter para `dict` comum facilita inspeção e possível serialização.

**Verificação e resultados:** Na execução de 2026-10-18: vocabulário lexical com 19.054 termos; 360.200 postings; comprimento médio processado de 165,82 tokens; tempo de construção medido em 3,2874 s. A medição começa depois da leitura do dataset e cobre tokenização, coleta do vocabulário pré-stemming, criação das posting lists e comprimentos. São valores de uma execução, sujeitos ao ambiente.

**Pendências:** Repetir/registrar o tempo no ambiente e condições usados para a entrega final, caso sejam diferentes.

### Etapa 4 — Busca lexical BM25

**Estado:** Implementada; parâmetros ainda não foram calibrados por experimento.

**Objetivo:** Ranquejar documentos lexicalmente usando postings, sem varrer todos os documentos a cada consulta.

**Implementação:** `search_bm25(query, top_k=10, k1=1.2, b=0.75)` valida `k1 > 0` e `0 <= b <= 1`, pré-processa a consulta da mesma forma que os documentos, considera termos repetidos da consulta uma única vez e acumula scores apenas para documentos nas posting lists dos termos. Retorna `doc_id` e `score`. A ordenação é score decrescente e, em empate, `doc_id` crescente.

Usa IDF suavizado positivo:

```text
IDF(t) = ln(1 + (N - df(t) + 0.5) / (df(t) + 0.5))

BM25(d,q) = soma_t IDF(t) * [tf(t,d) * (k1 + 1)] /
            [tf(t,d) + k1 * (1 - b + b * |d| / avgdl)]
```

**Decisões e justificativas:** `k1=1.2` e `b=0.75` são valores iniciais usuais. `k1` regula a saturação de TF e `b`, a normalização pelo comprimento. BM25 foi escolhido sobre TF-IDF com cosseno como baseline de ranking lexical: modela explicitamente saturação e comprimento e usa diretamente as posting lists. Isso é uma justificativa metodológica, não uma afirmação de superioridade empírica no NF-Corpus; a comparação dependerá dos experimentos.

**Medição de tempo:** `measure_bm25_query_time(split, top_k, repetitions)` mede tempo médio de busca para IDs presentes tanto nos qrels da partição escolhida quanto em `queries`. O padrão é `dev`. A medição exclui leitura do dataset e construção do índice. O conjunto `test` deve ser usado para avaliação final após fixar parâmetros.

**Verificação e resultados:** A busca BM25 foi executada no experimento da Etapa 5 para calcular rankings e NDCG@10 das condições avaliadas. Não foi registrada aqui uma medição média geral de tempo por consulta.

**Pendências:** Ajustar e fixar parâmetros usando `dev`; medir e registrar resultados; usar `test` apenas após congelar a configuração.

### Etapa 5 — Tolerância a erros

**Estado:** Implementada conforme o item 4 do PDF. Ajuste de parâmetros em `dev` e experimento final em dez consultas de `test` executados em 2026-10-18.

**Objetivo:** Corrigir termos desconhecidos usando k-gramas para gerar candidatos, Jaccard para filtrá-los e distância de edição para selecionar uma alternativa; medir a perda e recuperação de NDCG@10 no sistema lexical.

**Implementação:**

- `raw_vocabulary` é coletado dos tokens minúsculos do título e texto dos documentos antes de stemming/lematização, incluindo tokens que depois são removidos como stopwords. Esse é o dicionário usado para reconhecer termos conhecidos e corrigir OOV.
- `character_kgrams(term, k)` produz k-gramas de caracteres com padding de início/fim. `build_kgram_index` mapeia cada k-grama para os termos do vocabulário.
- Para cada token fora do vocabulário, `best_spelling_candidate` reúne candidatos pelo índice de k-gramas, filtra pelo coeficiente de Jaccard e escolhe a menor distância de Levenshtein. Empates são resolvidos por Jaccard maior e ordem lexical determinística.
- `correct_query` preserva tokens conhecidos e corrige somente OOV que encontrem candidato dentro dos limites.
- `make_oov_typo` cria uma única inserção, remoção, substituição ou transposição e garante que o token artificial não exista no vocabulário. A geração é independente dos parâmetros do corretor, para que todas as configurações de Dev sejam avaliadas sobre os mesmos erros.
- `tune_typo_parameters` aceita somente o conjunto `dev`. Com 10 casos fixos, compara 24 configurações: `k` em {2, 3, 4}, limiar Jaccard em {0,20; 0,30; 0,40; 0,50} e distância máxima em {1, 2}. O critério de seleção, em ordem, é acurácia na correção do token artificial, NDCG@10 médio corrigido, fração média recuperada nos casos com perda; desempates favorecem trigramas, limiar maior e distância menor. A tabela completa é gravada em `Experiments/typo_dev_tuning.csv`.
- Depois do ajuste em Dev, a configuração é congelada e `run_typo_experiment` executa as três condições (original, erro, corrigida) em exatamente 10 consultas de `test`, todas com NDCG@10 original maior que zero. NDCG usa ganho linear e IDCG calculado com todos os julgamentos da consulta.
- Para cada consulta com `NDCG10_original - NDCG10_erro > 0`, calcula `(NDCG10_corrigida - NDCG10_erro) / (NDCG10_original - NDCG10_erro)`. Caso contrário, registra “erro não reduziu o desempenho”. Os resultados são gravados em `Experiments/typo_experiment.csv`, com os campos mínimos do PDF e detalhes dos tokens e correções.

**Decisões e justificativas:** O ajuste em Dev é necessário para que parâmetros não sejam escolhidos olhando o desempenho de Test. O conjunto de Dev usa corrupções fixas e os mesmos 10 casos em todas as configurações, evitando selecionar um erro diferente para cada candidato. A acurácia da correção do token injetado tem prioridade para assegurar cobertura dos tipos de erro; NDCG corrigido e recuperação da perda também participam da decisão. A distância de Levenshtein pode representar uma transposição adjacente como duas operações; por isso o grid inclui distância máxima 2. As demais etapas de recuperação continuam usando as consultas originais, sem correção, conforme o PDF.

**Verificação e resultados:** O vocabulário de correção contém 26.475 termos; a construção do índice de k-gramas escolhido (k=3) levou 0,0967 s. O ajuste em Dev escolheu `k=3`, limiar Jaccard 0,30 e distância máxima 2, com acurácia de correção do token artificial 0,80, NDCG@10 médio corrigido 0,3932 e fração média recuperada 0,4887 em 9 casos degradados. O grid de 24 configurações foi salvo.

No Test foram avaliadas 10 consultas únicas: inserção (3), remoção (3), substituição (2) e troca de ordem (2). Todas tinham NDCG@10 original positivo e todos os termos artificiais eram OOV. O erro reduziu NDCG em 8 consultas; nas 2 restantes não houve perda, conforme o critério do PDF. A fração média recuperada entre os 8 casos com perda foi 0,75. O corretor recuperou exatamente o token injetado em 8/10 casos; nas consultas `PLAIN-1018` e `PLAIN-1088` selecionou outro termo candidato, e a NDCG corrigida permaneceu igual à NDCG com erro. A configuração de Test foi a escolhida em Dev, sem ajuste com Test.

O NLTK já tinha o recurso de stopwords instalado localmente. Uma execução anterior tentou baixá-lo e falhou ao validar o host; a inicialização foi ajustada para só solicitar download se o recurso local estiver ausente.

**Pendências:** O grid contém apenas os valores especificados acima; ampliar a busca de parâmetros, se necessário, deve continuar usando exclusivamente `dev`. Preservar `Experiments/typo_dev_tuning.csv` e `Experiments/typo_experiment.csv` nos artefatos finais.

### Etapa 6 — Recuperação semântica com sentence-BERT

**Estado:** Implementada e executada em 2026-10-18; embeddings pré-calculados e persistidos.

**Objetivo:** Codificar previamente cada documento com um bi-encoder pré-treinado, gerar embedding para cada consulta e ranquear documentos por similaridade, conforme a seção de recuperação semântica do PDF (item 5).

**Implementação:**

- A célula `semantic-precompute-and-search` em `Trabalho2.ipynb` carrega `sentence-transformers/all-MiniLM-L6-v2` com Sentence Transformers e seleciona CUDA quando disponível; na execução registrada, usou CPU.
- A entrada de cada documento é a concatenação de título e texto original. Não se aplica o stemming nem a remoção lexical de stopwords: o modelo semântico recebe texto natural.
- Os documentos são ordenados por ID antes da codificação para estabelecer alinhamento determinístico. A matriz de embeddings é `float32`, os vetores são L2-normalizados e seus IDs correspondentes são salvos separadamente.
- Os artefatos persistidos são `Experiments/semantic/all_minilm_l6_v2_document_embeddings.npy`, `Experiments/semantic/all_minilm_l6_v2_document_ids.json` e `Experiments/semantic/all_minilm_l6_v2_manifest.json`. O manifesto registra hash do conteúdo da coleção, dimensão, normalização, limite de tokens, truncamento, biblioteca, dispositivo e tempo de geração.
- O cache só é reutilizado se nome do modelo, hash dos textos, configuração de normalização, sequência de IDs e dimensões forem compatíveis; se a validação falhar, os vetores são recalculados.
- `search_semantic(query, top_k)` codifica a consulta original com o mesmo modelo e normalização, calcula produto interno contra a matriz e retorna os maiores scores. Como ambos os lados estão normalizados, o produto interno equivale à similaridade de cosseno. A busca é exaustiva sobre os 3.633 documentos.
- `measure_semantic_query_time` mede a codificação da consulta mais a busca, excluindo carga do modelo e geração dos vetores dos documentos.

**Decisões e justificativas:** O modelo `all-MiniLM-L6-v2` atende à opção sentence-BERT e ao uso de bi-encoder pré-treinado descritos no PDF. O pré-cálculo evita recodificar todos os documentos a cada consulta. A normalização permite usar produto interno como cosseno de forma eficiente e direta. A busca exaustiva é apropriada ao tamanho atual da coleção e está restrita à etapa semântica; não substitui o índice invertido na busca lexical. A concatenação de título e texto mantém o conteúdo documental completo disponível ao encoder, sujeito ao limite de entrada do modelo.

**Verificação e resultados:** Os embeddings foram efetivamente gerados em CPU com `sentence-transformers 6.1.0` e `torch 2.14.1+cu130`. Foram codificados 3.633 documentos em vetores de 384 dimensões; o arquivo NumPy ocupa 5.580.416 bytes. O modelo aceita até 256 tokens por entrada e 2.864 documentos excederam esse limite; o tokenizador/encoder trunca essas entradas ao comprimento máximo. A soma de norma de um vetor inspecionado foi 1,0. A chamada de `encode` para os documentos levou 112,90 s nesta execução (não inclui carregamento do modelo nem medição do comprimento dos textos). A busca de exemplo executou usando o cache validado. O tempo médio medido sobre as 324 consultas de `dev`, para `top_k=100` e incluindo embedding da consulta mais busca, foi 0,0928 s/consulta. Esses tempos dependem do ambiente e da carga da máquina. Nenhuma avaliação de qualidade semântica ou consulta de `test` foi executada nesta etapa.

**Pendências:** A fração alta de textos acima de 256 tokens deve ser explicitada no relatório; se o grupo decidir avaliar segmentação de documentos para preservar texto além do limite, essa configuração deverá ser ajustada usando `dev` e registrada separadamente. Os resultados da avaliação e comparação em `test` estão na Etapa 8.

### Etapa 7 — Reranking

**Estado:** Implementado e reexecutado em dev/test em 2026-10-26, com a fusão RRF corrigida (itens 8–9).

**Objetivo:** Usar o ranking híbrido como recuperação inicial, reordenar os 50 primeiros documentos com um cross-encoder e preservar a ordem relativa dos documentos que estavam abaixo da posição 50.

**Implementação:** `ri_project/reranking.py` implementa `CrossEncoderReranker` e `HybridRerankingRetriever`. O modelo `cross-encoder/ms-marco-MiniLM-L6-v2` (revisão `233902d25c440f23af6f7d6e94d2946bac0bee0a`, Sentence Transformers 6.1.0) pontua pares de consulta e documento formados pelo texto da consulta e pela concatenação de título e texto do documento. Os 50 documentos com maior score do cross-encoder compõem o início do ranking; os itens originalmente a partir da posição 51 mantêm sua ordem híbrida. A avaliação de `ri_project/evaluation.py` inclui Lexical, Semântico, Híbrido e Híbrido + reranking, grava `Experiments/results_reranking.csv` e métricas por consulta e agregadas. O manifesto em `Experiments/reranker_manifest.json` registra modelo, dispositivo, profundidade e comprimento de entrada.

**Decisões e justificativas:** O cross-encoder foi escolhido dentre os modelos sugeridos no enunciado por avaliar conjuntamente consulta e documento, permitindo interação entre os dois textos. A escolha foi examinada em `dev` antes de congelar o modelo para `test`. O limite configurado foi 512 tokens, batch size 32 e profundidade de reranking 50. Foi fixada a revisão do modelo para reproduzir os pesos usados na execução. O score final dos primeiros 50 é o score do cross-encoder; a cauda conserva o score RRF de origem e a ordem relativa, e o CSV preserva também `hybrid_score`, `reranker_score` e `hybrid_rank` para distinguir as fontes dos valores.

**Verificação e resultados:** Reexecução em GPU de 2026-10-26 após a correção dos candidatos RRF. Em dev (324 consultas), NDCG@10 passou de 0,3246 no híbrido para 0,3319 com reranking. O tempo médio total foi 0,3958 s/consulta, com 0,0194 s na recuperação híbrida e 0,3749 s na etapa cross-encoder. Foram pontuados 16.200 pares; 1.463 excederam 512 tokens antes do truncamento.

Em test, foram avaliadas 323 consultas por método, com rankings de 100 documentos por consulta:

| Método | P@10 | R@10 | MAP@10 | NDCG@10 | Tempo total médio (s/consulta) |
|---|---:|---:|---:|---:|---:|
| BM25 | 0,2390 | 0,1532 | 0,1218 | 0,3278 | 0,0011 |
| all-MiniLM-L6-v2 | 0,2433 | 0,1550 | 0,1108 | 0,3167 | 0,0171 |
| Híbrido RRF (`k=10`) | 0,2567 | 0,1698 | 0,1326 | 0,3505 | 0,0175 |
| Híbrido + reranking | 0,2511 | 0,1666 | 0,1388 | 0,3564 | 0,3433 |

Em test, a recuperação híbrida dentro do sistema reranqueado levou 0,0149 s/consulta e o cross-encoder 0,3270 s/consulta. Dos 16.150 pares pontuados, 1.537 excederam 512 tokens; maior comprimento antes de truncar: 2.389 tokens. Dispositivo: CUDA na RTX 4050 Laptop GPU. O manifesto real de dev está em `Experiments/reranker_dev_manifest.json`; o de test, em `Experiments/reranker_manifest.json`. Os contadores são reiniciados após dev.

A conferência independente em `Experiments/execution_2026-10-26/verification_retrieval.json` confirmou os quatro rankings, os qrels e as quatro métricas em todas as consultas, a equivalência da fusão com uma referência que usa apenas Top-100 de cada baseline e a preservação do conjunto Top-50 e da cauda no reranking.

**Pendências:** A análise por consulta do item 10 foi feita na Etapa 10. Discutir no relatório o truncamento de entrada e o custo/latência de inferência registrados nesta etapa.

### Etapa 8 — Avaliação lexical × semântica

**Estado:** Métricas, exportação dos resultados e comparação por consulta implementadas e executadas em `test` em 2026-10-19.

**Objetivo:** Avaliar BM25 e recuperação semântica nas mesmas consultas de teste com as métricas do item 6 do PDF e identificar os casos extremos de diferença de NDCG@10 requeridos pelo item 7 (PDF, pp. 3–4, itens 6–7).

**Implementação:**

- A célula `retrieval-evaluation-test` calcula Precision@10, Recall@10, AP@10 e NDCG@10 por consulta e as médias macro. Relevância binária é `score > 0` para P/R/AP; Recall e AP usam como denominador todos os documentos relevantes da consulta nos qrels. NDCG preserva os graus dos qrels, usa ganho linear e calcula o ideal sobre todos os julgamentos da consulta.
- BM25 foi avaliado com `k1=1.2`, `b=0.75`, e semântica com `sentence-transformers/all-MiniLM-L6-v2`, vetores de 384 dimensões e cosseno. São as configurações congeladas para esta avaliação; não houve ajuste olhando para os resultados de `test`.
- Foram usadas as 323 consultas que têm qrels em `qrels/test.tsv` e registro em `queries.jsonl`, igual para os dois métodos. Os rankings exportados têm profundidade 100. BM25 pontua somente candidatos vindos das posting lists. Se houver menos de 100 candidatos, documentos não pontuados são completados com score zero e desempate crescente por ID; a complementação inspeciona apenas os IDs necessários à profundidade de exportação e não calcula BM25 para o corpus inteiro.
- Os arquivos padronizados `Experiments/results_lexical.csv` e `Experiments/results_semantic.csv` têm `query_id`, `doc_id`, `rank`, `score` e `relevance`. `Experiments/metrics_per_query.csv` tem `query_id`, `model`, `P10`, `R10`, `AP10` e `NDCG10`; `Experiments/metrics_summary.csv` guarda médias, número de consultas e tempo por consulta.
- `Experiments/comparison_cases_summary.csv` guarda as três consultas com maior ganho semântico e as três com maior ganho lexical por ΔNDCG@10 (semântico menos lexical). `Experiments/comparison_cases_top5.csv` apresenta os Top5 dos dois métodos por caso, com texto da consulta/documento, scores, graus de relevância e termos da consulta que coincidem após pré-processamento lexical. A análise interpretativa está em `Experiments/comparison_cases_analysis.md`. Duas células finais no notebook imprimem separadamente os três casos com vantagem semântica e os três com vantagem lexical, mostrando para cada consulta os Top5 de ambos os métodos e marcando cada documento como relevante (com grau) ou não relevante.

**Decisões e justificativas:** As duas condições compartilham exatamente a mesma partição e consulta, de modo que a comparação seja pareada. Os scores BM25 e cosseno têm escalas diferentes e não são comparados numericamente entre si; a comparação usa métricas de ranking e ΔNDCG@10. O desempate lexical de score zero por ID é determinístico e usado apenas para cumprir a profundidade Top100 quando a lista de postings não fornece candidatos suficientes; os scores positivos e sua ordem vêm exclusivamente do BM25 sobre o índice invertido.

**Verificação e resultados:** Foram avaliadas 323 consultas por método. Médias em `test`:

| Método | P@10 | R@10 | MAP@10 | NDCG@10 | Tempo médio por consulta |
|---|---:|---:|---:|---:|---:|
| BM25 (`k1=1.2`, `b=0.75`) | 0,2390 | 0,1532 | 0,1218 | 0,3278 | 0,0006 s |
| all-MiniLM-L6-v2 | 0,2433 | 0,1550 | 0,1108 | 0,3167 | 0,0617 s |

A média de `AP10` em `Experiments/metrics_per_query.csv` coincide com `MAP10` em `Experiments/metrics_summary.csv` para ambos os métodos. Cada arquivo de ranking contém 32.300 linhas (100 por consulta). O semântico obteve P@10 e R@10 ligeiramente maiores; BM25 obteve MAP@10 e NDCG@10 maiores. Os tempos foram medidos no mesmo processo após carregar o modelo e aquecer a codificação semântica; incluem busca e embedding da consulta para semântica, mas não incluem carga do modelo nem geração dos embeddings documentais. Os scores não foram interpretados como comparáveis entre métodos.

A diferença semântica − lexical de NDCG@10 foi: `PLAIN-307` +0,6714, `PLAIN-3085` +0,5706 e `PLAIN-1537` +0,5670. A maior vantagem lexical ocorreu em `PLAIN-1473` (−1,0000), seguida por `PLAIN-2081` (−1,0000) e `PLAIN-1288` (−0,7601). Consultas, Top5, julgamentos e evidências textuais estão nos CSVs acima; interpretação dos casos está no arquivo Markdown de análise.

**Pendências:** A avaliação híbrida RRF e o reranking foram executados posteriormente e estão registrados nas Etapas 9 e 7; a análise por consulta está na Etapa 10. O pipeline RAG foi implementado na Etapa 11; sua execução real e os experimentos seguintes continuam pendentes. A escolha de modelo semântico foi fixada previamente por solicitação do projeto; caso o grupo queira comparar modelos alternativos, essa seleção deve ser feita exclusivamente em `dev`, sem alterar esta configuração com base no `test`.

## Correções feitas no esboço anterior

- O esboço tinha funções auxiliares indentadas dentro de `preprocess` e depois de um `return`, portanto não seriam definidas nem chamadas na execução normal. O trecho foi substituído por uma função direta, com imports explícitos e inicialização única de stopwords e stemmer.
- A instalação de dependências foi ajustada para usar a magic explícita `%pip install` do Jupyter.
- Foram adicionadas células separadas e documentadas para leitura do dataset e construção do índice.

## Estruturas disponíveis

- `documents`: documentos completos indexados por ID.
- `queries`: consultas indexadas por ID.
- `qrels`: qrels separados em `train`, `dev` e `test`.
- `qrels_all`: qrels concatenados.
- `tokenize(text)` e `raw_vocabulary`: tokens normalizados e vocabulário de correção antes do stemming.
- `preprocess(text)`: processamento textual lexical comum a documentos e consultas.
- `inverted_index`: postings com frequência por termo e documento.
- `document_lengths`, `num_documents` e `avg_document_length`: estatísticas usadas pelo BM25.
- `num_postings` e `index_build_time_seconds`: total de pares termo-documento e tempo de construção.
- `search_bm25(query, top_k, k1, b)`: ranking lexical baseado em postings e estatísticas do índice.
- `semantic_document_embeddings` e `semantic_document_ids`: vetores densos pré-calculados e IDs alinhados por linha.
- `search_semantic(query, top_k)`: busca por cosseno usando embeddings pré-calculados.
- `measure_semantic_query_time(split, top_k, repetitions)`: mede latência média de embedding da consulta mais busca.
- `Experiments/metrics_per_query.csv` e `Experiments/metrics_summary.csv`: resultados quantitativos de teste lexical e semântico.
- `Experiments/results_lexical.csv` e `Experiments/results_semantic.csv`: rankings Top100 com scores e julgamentos.
- `Experiments/comparison_cases_summary.csv`, `Experiments/comparison_cases_top5.csv` e `Experiments/comparison_cases_analysis.md`: comparação dos seis casos extremos de NDCG@10.
- `measure_bm25_query_time(split, top_k, repetitions)`: tempo médio de busca por consulta numa partição de qrels.
- `kgram_index`, `character_kgrams`, `jaccard_coefficient`, `edit_distance`, `correct_query` e `run_typo_experiment`: geração, filtragem, seleção de correção e experimento OOV.
- `Experiments/typo_dev_tuning.csv`: comparação das 24 configurações avaliadas em Dev.
- `Experiments/typo_experiment.csv`: resultados por consulta das condições original, com erro e corrigida em Test.

## Registro de execução das notas

- `Notas_Projeto_RI.txt` foi convertido para este arquivo Markdown, mantendo o conteúdo técnico e reorganizando cada etapa com os campos **estado**, **objetivo**, **implementação**, **decisões e justificativas**, **verificação e resultados** e **pendências**.
- Em 2026-10-18, as células do notebook foram executadas para o ajuste de parâmetros em Dev e o experimento lexical de tolerância a erros; resultados e limitações estão registrados nas Etapas 1–5. Não foram executados os demais experimentos.


## Registro de execução — 2026-10-18

A etapa 6 foi implementada e executada. Os embeddings e manifesto estão salvos sob `Experiments/semantic/`; a configuração e os resultados confirmados estão descritos na etapa. O cache local do modelo usado para a execução não faz parte dos artefatos do projeto; em uma primeira execução em outro ambiente será necessário que o modelo esteja disponível localmente ou que o Hugging Face possa ser acessado.


## Registro de execução — 2026-10-19

A avaliação dos itens 6–7 do PDF foi executada após congelar BM25 padrão (`k1=1.2`, `b=0.75`) e o modelo semântico solicitado. Os artefatos CSV e a análise por consulta estão listados na Etapa 8. Não houve ajuste de parâmetros com base em `test`.


## Registro de organização dos artefatos — 2026-10-19

Todos os arquivos de saída dos experimentos, incluindo rankings, métricas, resultados de tolerância a erros, comparação por consulta e embeddings pré-calculados, foram centralizados em `Experiments/`. O notebook cria a pasta ao carregar o dataset e escreve os novos resultados nesse local. A pasta está listada no `.gitignore` para que os artefatos locais não sejam enviados ao GitHub; os arquivos podem ser regenerados executando o notebook.

### Registro de modularização — 2026-10-19

**Estado:** Implementação reorganizada; experimentos não foram reexecutados nesta alteração.

**Objetivo:** Separar a implementação do notebook em módulos Python para facilitar manutenção e reutilização, mantendo o notebook como fluxo principal do trabalho.

**Implementação:** Criado o pacote `ri_project/` com módulos para configuração (`config.py`), leitura do NF-Corpus (`data.py`), pré-processamento (`preprocessing.py`), índice invertido (`indexing.py`), BM25 (`bm25.py`), tolerância a erros e ajuste em Dev (`typo.py`), embeddings/busca semântica (`semantic.py`) e métricas/comparações (`evaluation.py`). O notebook foi reduzido a imports, chamadas sequenciais dessas APIs, gravação dos resultados e apresentação das tabelas/consultas. Os caminhos dos artefatos continuam centralizados em `Experiments/`. O `README.md` documenta a estrutura, dependências e ordem de execução.

**Decisões e justificativas:** As funções que dependem de estado do índice ou dos embeddings foram agrupadas em `BM25Retriever` e `SemanticRetriever`; os fluxos de tolerância a erros e avaliação recebem essas dependências explicitamente. Isso torna os limites entre os métodos visíveis e reduz a dependência de variáveis globais do notebook. A sequência experimental existente continua declarada no notebook, que serve como interface didática e executável. `Experiments/` permanece ignorada pelo Git conforme a organização anterior.

**Verificação e resultados:** `python3 -m py_compile ri_project/*.py` concluiu sem erros. As células de código do notebook foram compiladas individualmente e a estrutura JSON do notebook foi lida com sucesso. Nenhum experimento foi executado; portanto, não há novos tempos, métricas ou confirmação de equivalência numérica nesta etapa.

**Pendências:** Executar o notebook integralmente em um ambiente com NF-Corpus, dependências e modelo disponíveis para confirmar o fluxo de ponta a ponta e regenerar os artefatos locais, caso necessário. Na data deste registro, o item de reranking ainda estava pendente; a implementação posterior está registrada na Etapa 7.


### Etapa 9 — Busca híbrida com Reciprocal Rank Fusion

**Estado:** Ajuste em dev e avaliação em test refeitos em 2026-10-26 após corrigir `complete_ranking`.

**Objetivo:** Combinar os 100 primeiros resultados de BM25 e sentence-BERT por RRF, escolher `k` usando apenas `dev` e comparar lexical, semântico e híbrido em `test`.

**Implementação:** `ri_project/hybrid.py` contém `reciprocal_rank_fusion`, `complete_ranking` e `HybridRRFRetriever`. A fórmula é `1 / (k + rank)`, com rank iniciando em 1 e sem contribuição de uma lista em que o documento não aparece. `tune_rrf_k` em `ri_project/evaluation.py` comparou `k ∈ {10, 60, 100}` sobre as 324 consultas de `dev`, usando Top-100 para cada método, calculou NDCG@10 por consulta e média macro e selecionou o maior valor. Em empate o código escolhe o menor `k`. Os resultados de ajuste estão em `Experiments/rrf_dev_tuning.csv` e `Experiments/rrf_dev_tuning_per_query.csv`. O parâmetro selecionado é congelado para `test`. O ranking híbrido padronizado fica em `Experiments/results_hybrid.csv`; `metrics_per_query.csv` e `metrics_summary.csv` incluem a linha híbrida.

**Decisões e justificativas:** Quando a lista já tem a profundidade solicitada, `complete_ranking` retorna imediatamente; a complementação ocorre somente em listas curtas. Três testes de regressão cobrem lista completa sem varrer IDs, preenchimento curto e limite dos candidatos na fusão. Os valores de `k`, o critério NDCG@10 e a profundidade de candidatos seguem o item 8. Quando BM25 retorna menos de 100 candidatos positivos, sua lista é completada com documentos de score zero em ordem de ID, consistente com o Top-100 lexical usado na avaliação; documentos fora do Top-100 completado recebem contribuição apenas se estiverem no ranking semântico.

**Verificação e resultados:** Refeito em 2026-10-26 com listas limitadas a 100 documentos. NDCG@10 médio em dev: `k=10` 0,324595, `k=60` 0,318746 e `k=100` 0,319253. Selecionado `k=10`, exclusivamente com as 324 consultas de dev. Em test, sobre 323 consultas, o híbrido obteve P@10 0,2567, R@10 0,1698, MAP@10 0,1326, NDCG@10 0,3505 e tempo médio de 0,0175 s/consulta. Cada ranking contém 100 documentos. Os valores anteriores à correção foram substituídos; os baselines não dependem da função corrigida.

**Pendências:** Item 8 concluído; análise do reranking na Etapa 10, geração RAG na Etapa 12 e avaliação/análise implementadas nas Etapas 13–14. As notas e revisões humanas permanecem pendentes.

### Etapa 10 — Análise por consulta do reranking

**Estado:** Análise reexecutada em test em 2026-10-26 após a correção do RRF, conforme item 10.

**Objetivo:** Examinar os dois maiores ganhos de NDCG@10 após reranking, um caso com pouca ou nenhuma alteração e a maior piora, mostrando documentos movimentados e explicações apoiadas nos títulos, posições e qrels.

**Implementação:** `build_reranking_analysis` em `ri_project/evaluation.py` compara, por consulta, os rankings híbrido e reranqueado da mesma avaliação. Seleciona os casos pelo delta `NDCG@10 reranqueado − NDCG@10 híbrido`; para o caso mais próximo de zero, empates são desfeitos pelo menor número de documentos cuja mudança de posição afeta o Top-10. A função exporta `Experiments/reranking_analysis_cases.csv`, `Experiments/reranking_analysis_movements.csv` e `Experiments/reranking_analysis_top10.csv`. A célula da seção de análise em `Trabalho2.ipynb` reproduz a seleção e exibe as tabelas. A interpretação textual dos quatro casos está em `Experiments/reranking_analysis.md`.

**Decisões e justificativas:** A seleção e as conclusões usam apenas resultados e julgamentos do conjunto `test`, sem retunar configurações. Delta positivo indica ganho. Para pouca alteração, foi escolhido delta exatamente zero e, entre casos equivalentes, o menor número de documentos cuja mudança de posição afeta o Top-10; o ID da consulta resolve empates restantes; graus iguais podem trocar de posição sem alterar o NDCG. A discussão de causas é apresentada como hipótese apoiada nos títulos e posições, sem inferir mecanismos internos do cross-encoder a partir de seus scores.

**Verificação e resultados:** Refeito em 2026-10-26 nas 323 consultas pareadas. Maiores ganhos: `PLAIN-1320` (“Harvard Physicians’ Study II”), 0,0000 → 0,5617 (Δ +0,5617), com MED-2760, MED-2758 e MED-2761 (grau 1) promovidos de 11, 12 e 19 para 1, 3 e 6; e `PLAIN-1473` (“kohlrabi”), 0,6309 → 1,0000 (Δ +0,3691), com MED-4455 (grau 1) de 2 para 1. O caso estável agora é `PLAIN-1557` (“magnesium”), 0,315163 → 0,315163: MED-3918 e MED-4315 (ambos grau 1) trocam as posições 2 e 3, e MED-4313 (grau 1) permanece em 9. O vetor de graus do Top-10 fica idêntico, apesar de oito documentos mudarem de posição envolvendo essa janela. A maior queda continua em `PLAIN-2540` (“Does Cholesterol Size Matter?”), 0,6153 → 0,3031 (Δ −0,3123); MED-1890 (grau 2) cai de 1 para 15 e MED-1205 (sem julgamento positivo) sobe de 21 para 1. Os CSVs e o Markdown interpretativo foram atualizados com as posições atuais. Grau 0 na exportação pode representar ausência de julgamento; não demonstra irrelevância absoluta.

**Pendências:** Item 10 concluído; geração RAG executada na Etapa 12. Métricas de contexto calculadas e fluxos de avaliação/análise implementados nas Etapas 13–14, aguardando preenchimento humano.

### Etapa 11 — Pipeline de Retrieval-Augmented Generation (RAG)

**Estado:** Implementado e executado com NF-Corpus e Qwen na GPU local em 2026-10-26; experimento pareado registrado na Etapa 12.

**Objetivo:** Implementar o item 11: consulta original → recuperação híbrida → reranking → seleção de contexto → LLM → resposta, mantendo cada registro de `corpus.jsonl` como unidade de contexto.

**Implementação:** `ri_project/rag.py` oferece `RAGConfig`, `select_context`, `build_rag_messages`, `LocalCausalGenerator` e `RAGPipeline`. O pipeline reutiliza `HybridRerankingRetriever`, solicita profundidade 100 para preservar o reranking dos 50 candidatos e seleciona os cinco primeiros documentos na ordem final. Título, texto integral e `doc_id` são enviados à LLM, sem chunking, sem consulta aos graus dos qrels e sem correção da pergunta. O prompt exige fontes `[doc_id]`, resposta baseada somente nas evidências e declaração de insuficiência quando aplicável. O tokenizador aplica o chat template do modelo e mede o prompt completo, sem truncamento, reservando espaço para a saída; estouro de janela gera erro explícito. O JSON de saída preserva fontes, scores/posições de recuperação, mensagens, prompt formatado, resposta, tokens, tempos, RRF, manifesto do reranker e manifesto da LLM. A seção de RAG em `Trabalho2.ipynb` demonstra a geração em uma consulta determinística de `dev` e a exportação para `Experiments/rag/`. O README documenta instalação, configuração e execução.

**Decisões e justificativas:** Modelo inicial `Qwen/Qwen2.5-1.5B-Instruct` via Transformers, executado localmente em CUDA ou CPU, permitindo uso sem API paga. Trata-se de escolha de implementação, não de conclusão experimental em test. A documentação primária consultada foi o [model card do Qwen](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct) e a [documentação de geração do Transformers](https://huggingface.co/docs/transformers/main_classes/text_generation). A configuração inicial usa inglês (idioma da coleção), seed 42, temperatura 0 com decodificação greedy e até 512 novos tokens. Na implementação inicial, Top-5 e instruções de fidelidade anteciparam a interface do item 12, cujo experimento foi executado na Etapa 12. O manifesto registra revisão resolvida, versões, dtype, janela e CPU/GPU observados na execução. A revisão inicial era `main`; na execução da Etapa 12 foi identificada e fixada a revisão `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`, utilizada nos dois notebooks e no gerador padrão. Não há cobrança de API; energia e hardware não são estimados. O gerador local importa dependências pesadas apenas na inicialização; a lógica de contexto pode ser verificada isoladamente.

**Verificação e resultados:** Seis testes `unittest` passaram com componentes simulados: preservação do Top-5 e da profundidade de reranking, exportação JSON, conteúdo integral e instruções do prompt, rejeição de contexto inválido/configuração inválida/consulta vazia, rejeição de estouro da janela antes de gerar e verificação do adaptador de geração (decodificação somente da resposta, seed, limite de saída e seleção greedy/amostragem). Todos os módulos Python e células de código do notebook foram compilados em memória, e seu JSON foi lido com sucesso. Estes testes não validam qualidade das respostas nem execução real de Transformers. Na verificação inicial dessa etapa, o ambiente ainda não continha o dataset, PyTorch, Transformers, sentence-transformers, pandas ou NLTK; nenhum peso havia sido baixado e nenhuma métrica, latência ou resposta real de RAG havia sido medida. O usuário inseriu posteriormente o dataset e a execução real está registrada na Etapa 12.

A execução real da demonstração em dev utilizou `PLAIN-1`, com 1.877 tokens de entrada, 292 tokens de saída, 15,4734 s de geração e 16,3727 s no pipeline completo. A saída auditável está em `Experiments/rag/PLAIN-1.json`. Foi utilizada a RTX 4050 Laptop GPU com BF16, revisão fixada da Qwen, janela de 32.768 tokens e documentos integrais. Esses números descrevem uma consulta de dev e não substituem a avaliação humana das respostas.

**Pendências:** Pipeline e geração executados. Etapas 13–14 implementam métricas de contexto, avaliação humana e análise de falhas; faltam as notas e a confirmação dos casos pelos integrantes.


### Etapa 12 — Experimento sem RAG × com RAG

**Estado:** Implementado, executado e verificado em 2026-10-26: vinte respostas concluídas em GPU local. `Experiments/rag_experiment/run_status.json` registra `complete`, com 20/20 respostas.

**Objetivo:** Executar o item 12 com as dez perguntas fixas de test, nas condições A (sem documentos) e B (Top-5 após reranking), usando a mesma LLM e parâmetros.

**Implementação:** `ri_project/rag_experiment.py` reproduz o sorteio de seed 42 entre as 42 perguntas elegíveis e confere a lista oficial. O notebook principal executa as duas condições em conversas independentes, reinicia a seed por chamada e usa os documentos integrais da ordem final. Os vinte JSONs são salvos em `responses/`, e as exportações finais são `answers.jsonl`, `answers.csv`, `comparison.md` e `rag_experiment.csv`. Os hashes SHA-256 de corpus, consultas e qrels de dev/test foram incluídos também na exportação do notebook, atendendo ao contrato de `scripts/verify_rag_experiment.py`. O arquivo padronizado contém notas humanas em branco e rastreabilidade de A como N/A. A retomada exige a mesma assinatura de contextos, configuração e manifesto; registros divergentes são regenerados.

**Decisões e justificativas:** Foi utilizado o ambiente virtual local, onde o PyTorch acessou a NVIDIA GeForce RTX 4050 Laptop GPU (6 GB) ao executar fora do sandbox que bloqueava CUDA/NVML. Não foi necessário usar o kernel Colab. A Qwen2.5-1.5B-Instruct usou BF16, revisão `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`, inglês, seed 42, temperatura configurada zero/greedy, máximo de 512 novos tokens e penalidade de repetição 1,1 herdada dos pesos. Hardware, dtype, janela e kwargs efetivos estão em cada resposta. O fluxo completo do notebook coube no ambiente local; a CLI por processos separados continua disponível para ambientes com menos memória. Não houve ajuste do prompt, do modelo ou de parâmetros com base nas respostas de test. Não houve cobrança de API; energia/hardware não foram estimados.

O registro anterior mencionava geração CPU em outro ambiente e arquivos `environment_cpu.txt` que não estavam presentes nesta cópia do projeto. Esse andamento não foi tratado como execução confirmada aqui e nenhuma resposta CPU foi misturada ao novo experimento GPU. `RAG_Colab.ipynb` foi regenerado com o código e contextos atuais, protegidos por SHA-256; sua execução na T4 não foi realizada nesta revisão.

**Verificação e resultados:** Foram confirmadas 3.633 unidades documentais e a seleção PLAIN-68, PLAIN-2450, PLAIN-153, PLAIN-2850, PLAIN-2750, PLAIN-270, PLAIN-2470, PLAIN-2430, PLAIN-238 e PLAIN-3342. Foram reutilizados embeddings documentais validados (384 dimensões); não foi medida uma nova geração dos embeddings. O ajuste RRF em dev foi refeito com a correção do Top-100, selecionando k=10 (ver Etapa 9). A geração utilizou Python 3.14.4, PyTorch 2.14.1+cu130, Transformers 5.20.1, sentence-transformers 6.1.0, Accelerate 1.15.0 e quatro threads PyTorch.

O verificador real confirmou os dez pares A/B com o mesmo modelo e parâmetros, nenhum documento fornecido em A, exatamente cinco documentos únicos em B e título/texto idênticos ao corpus. Os hashes do dataset foram conferidos. A janela da LLM é 32.768 tokens; o maior prompt B teve 2.979 tokens, sem truncar ou remover documentos, e nenhuma saída atingiu o limite de 512 novos tokens. Resumo medido:

| Condição | Respostas | Entrada média (tokens) | Saída média (tokens) | Geração média (s) |
|---|---:|---:|---:|---:|
| A — sem RAG | 10 | 25,3 | 230,3 | 12,3647 |
| B — com RAG | 10 | 2335,2 | 201,8 | 10,3087 |

Os tempos são descritivos e devem ser interpretados junto do comprimento das saídas. Nenhuma conclusão sobre qualidade foi atribuída por esse verificador. `Experiments/rag_experiment/verification.json` contém o resumo completo, e `comparison.md` reúne as respostas para avaliação humana. A conferência independente da recuperação está em `Experiments/execution_2026-10-26/verification_retrieval.json`.

**Pendências:** Métricas de contexto e infraestrutura dos itens 13–14 implementadas nas etapas seguintes. Preencher notas/justificativas e revisar os três casos para obter a comparação humana e o relatório final. Não há geração CPU ou execução Colab em andamento nesta sessão.


## Registro de revisão e execução — 2026-10-26

**Estado:** Fluxo completo de cálculos do notebook concluído na GPU local; instalação de dependências da célula inicial omitida porque o ambiente virtual já estava disponível. As versões efetivas foram registradas e as instruções de instalação foram alinhadas às versões usadas.

**Implementação e decisões:** Corrigido o limite Top-100 de `complete_ranking`, incluídos três testes de regressão, exportados os hashes do dataset na seção RAG e preservado o notebook anterior em `Experiments/execution_2026-10-26/Trabalho2.before_execution.ipynb`. Uma execução inicial foi interrompida antes da avaliação final para aplicar a correção. As saídas do notebook principal foram atualizadas com a execução completa posterior, e a exportação futura do manifesto de dev foi adicionada à sua célula. O manifesto desta execução foi exportado da saída efetivamente observada. O notebook Colab foi regenerado com os contextos corrigidos, sem ser executado remotamente.

**Verificação e resultados:** Treze testes unitários passaram. A execução completa usou nbclient no Python do projeto e terminou com status `complete`; notebook executado, log, ambiente e status estão em `Experiments/execution_2026-10-26/`. Uma referência independente confirmou RRF, limites, qrels e todas as métricas dos quatro métodos para as 323 consultas. O verificador RAG confirmou vinte respostas e dez pares íntegros. Não foram atribuídas notas humanas automaticamente.

**Pendências atualizadas:** Métricas de contexto e código de avaliação/análise implementados nas Etapas 13–14. Aguardam-se as notas e revisões humanas, além da redação do relatório final.


### Etapa 13 — Avaliação do contexto e das respostas RAG

**Estado:** Implementado e executado em 2026-10-26 sobre os contextos e vinte respostas já congelados. Métricas de recuperação calculadas; avaliação humana real ainda sem notas.

**Objetivo:** Aplicar o item 13: Recall@5, Precision@5 e Hit@5 dos documentos que chegaram à LLM, e avaliação humana estruturada de Fidelidade, Completude, Utilidade e Rastreabilidade nas condições A/B.

**Implementação:** `ri_project/rag_evaluation.py` oferece leitura/validação dos artefatos, métricas binárias de contexto, preparação das fichas e consolidação. A CLI `python -m ri_project.rag_evaluation prepare` e a nova seção do notebook escrevem em `Experiments/rag_experiment/evaluation/`. `context_metrics_per_query.csv` registra a ordem e IDs reais dos cinco documentos, quantidade de relevantes e P5/R5/Hit5 por consulta; `context_metrics_summary.csv` contém médias macro. A carga confere hashes do corpus, consultas e qrels, identidade dos pares A/B, fontes integrais e correspondência do contexto com o Top-5 final. Relevância é grau > 0 e Recall usa todos os relevantes da consulta nos qrels. Graus ausentes são tratados como zero para as métricas, sem presumir irrelevância absoluta.

O caderno `human_review.md` apresenta as vinte respostas em ordem embaralhada com seed 42 e os mesmos Top-5 de referência para avaliar ambas as condições. `blind_key.json` preserva o vínculo com consulta e condição, separado das fichas `ratings_reviewer_1.csv` e `ratings_reviewer_2.csv`. As fichas não são sobrescritas ao preparar novamente. A consolidação aceita rascunhos e valida IDs únicos, notas 0–2 e justificativas para cada nota aplicável. Campos vazios são pendências, nunca zeros. São exigidas 70 notas aplicáveis por avaliador: três critérios em vinte respostas e Rastreabilidade nas dez respostas B. A rastreabilidade provisória da etapa mascarada é descartada em A, exportada como N/A e excluída da média. A fidelidade de A é julgada contra os mesmos documentos usados em B, conforme o enunciado, mesmo sem terem sido enviados à LLM em A.

`human_scores_per_answer.csv` mantém notas por avaliador; `human_scores_summary.csv` informa médias, aplicabilidade, contagens e completude; `human_agreement.csv` registra concordância exata e kappa de Cohen com pesos quadráticos entre dois avaliadores, por condição/critério. Kappa com marginais sem variação fica indefinido. Com uma ficha completa ela pode ser usada como final; com duas completas é necessária escolha explícita por `--final-reviewer`. `human_evaluation_report.md` apresenta as médias e diferenças somente quando há uma avaliação completa escolhida. `quality_comparison_per_query.csv` registra deltas dos critérios comuns. As notas finais inteiras e justificativas são incorporadas ao `rag_experiment.csv` do item 15. A exportação da geração preserva essas notas somente se consulta, prompt, resposta e contexto permanecem idênticos; mudanças invalidam as anotações relacionadas.

**Decisões e justificativas:** O professor exige avaliação humana, portanto nenhum LLM ou heurística atribui notas de qualidade. O mascaramento e o segundo avaliador são recomendações do enunciado; seed 42 para embaralhar é uma decisão local. O estilo e as citações podem sugerir a condição, limitação registrada no guia. Não são calculadas médias entre avaliadores para criar notas fracionárias, incompatíveis com a escala obrigatória. Os hashes ligam fichas, respostas e contextos e rejeitam reutilização em outro experimento. O preparo é uma etapa de análise sobre arquivos existentes, sem novo ajuste de modelos, prompts ou rankings em test.

**Verificação e resultados:** As quatro células novas de avaliação/análise foram executadas em CPU, reutilizando as vinte respostas GPU e sem inferência adicional. Os 23 testes unitários passaram, incluindo métricas com denominador total, rascunhos sem conversão em zero, exclusão de N/A, validação de notas/justificativas, concordância degenerada, escolha final de avaliador e preservação/invalidação das notas na reexportação. Os testes da consolidação completa usam dados sintéticos em diretórios temporários, não avaliações humanas reais.

Médias macro nas dez perguntas oficiais: **P@5 0,4600, R@5 0,107495 e Hit@5 0,9000**. Foram enviados 50 documentos (contando repetições entre consultas), dos quais 23 têm grau positivo; nove consultas receberam pelo menos um relevante. Esses valores são do subconjunto de dez consultas e não da avaliação Top-10 das 323 consultas. Recall baixo, isoladamente, não demonstra falha: há muitos relevantes por consulta e o contexto tem somente cinco unidades.

| Consulta | Relevantes no contexto | Total relevante | P@5 | R@5 | Hit@5 |
|---|---:|---:|---:|---:|---:|
| PLAIN-68 | 1 | 55 | 0,2000 | 0,0182 | 1 |
| PLAIN-2450 | 2 | 38 | 0,4000 | 0,0526 | 1 |
| PLAIN-153 | 5 | 47 | 1,0000 | 0,1064 | 1 |
| PLAIN-2850 | 0 | 9 | 0,0000 | 0,0000 | 0 |
| PLAIN-2750 | 4 | 59 | 0,8000 | 0,0678 | 1 |
| PLAIN-270 | 2 | 52 | 0,4000 | 0,0385 | 1 |
| PLAIN-2470 | 3 | 73 | 0,6000 | 0,0411 | 1 |
| PLAIN-2430 | 3 | 46 | 0,6000 | 0,0652 | 1 |
| PLAIN-238 | 1 | 54 | 0,2000 | 0,0185 | 1 |
| PLAIN-3342 | 2 | 3 | 0,4000 | 0,6667 | 1 |

No estado real atual, `human_evaluation_status.json` registra `pending_human_ratings` e zero notas aplicáveis preenchidas. Não existem médias finais de qualidade ou notas humanas fabricadas. O notebook executado das novas seções e seu estado estão em `evaluation_sections.executed.ipynb` e `notebook_execution_status.json`, dentro da pasta de avaliação.

**Pendências:** Os integrantes devem preencher pelo menos uma ficha real e suas justificativas; recomenda-se a segunda ficha independente. Executar `aggregate` após o preenchimento para obter as médias finais, a discussão e os campos de avaliação no CSV padronizado.

### Etapa 14 — Análise de falhas ponta a ponta

**Estado:** Diagnóstico automático implementado e executado para as dez consultas; três casos preliminares discutidos com apoio de IA. Classificação e relatório final dependem de notas e revisões humanas reais.

**Objetivo:** Aplicar o item 14, examinando uma melhora clara, pouca diferença e ausência de ajuda/piora, com investigação da chegada da informação ao contexto e dos estágios envolvidos.

**Implementação:** `ri_project/rag_analysis.py` gera `stage_metrics_per_query.csv`, `stage_document_evidence.csv` e `failure_analysis_dossiers.md`. São preservados os textos dos documentos, graus e presença nos qrels, posições lexical/semântica/RRF/reranking e participação no contexto real. A tabela inclui relevantes que não foram recuperados, com posições ausentes. O preparo confere a fusão com os Top-100 dos baselines, o conjunto Top-50/cauda e o ranking final com os contextos efetivamente enviados. Hashes dos rankings e diagnósticos ficam em `evaluation_manifest.json`.

Após consolidar as notas humanas, `failure_case_selection.json` sugere consultas distintas pela diferença B−A da média de Fidelidade, Completude e Utilidade, com pesos iguais. A maior diferença positiva sugere melhora; a menor não positiva sugere ausência de ajuda. Para pouca diferença, é permitido no máximo um ponto de mudança absoluta total entre os três critérios, equivalente a até 1/3 na diferença média; isso evita chamar de estáveis grandes ganhos e perdas que se cancelam. Entre candidatos é escolhido o menor delta absoluto e depois o ID. Ausência de uma categoria é registrada explicitamente. Esses critérios auxiliam a seleção e não são exigências do professor.

O grupo confirma ou altera os casos em `case_reviews.json`, com motivo da seleção, comparação das respostas, indicação de informação presente/ausente/parcial, estágio principal e trechos literais de documentos. `python -m ri_project.rag_evaluation analyze` valida três consultas distintas e as categorias do item 14, notas humanas completas, justificativas e evidências. A alegação de informação recebida precisa incluir trecho de um documento do contexto real. No caso sem ajuda, deve ser identificado e explicado o estágio investigado. O relatório `failure_analysis.md` inclui respostas, notas A/B, posições e evidências e somente é publicado com revisões completas. A validação verifica existência literal e envio das fontes; o suporte semântico das afirmações continua sendo responsabilidade do avaliador. Alterações nas notas invalidam o relatório derivado anterior, preservado como histórico.

**Decisões e justificativas:** Relevância nos qrels não basta para concluir que a informação necessária chegou ou que a geração foi fiel. O diagnóstico permite inspecionar o texto e rastrear um documento desde os candidatos até o corte Top-5. Nenhum documento é inserido manualmente no contexto com base em qrels. A média auxiliar exclui Rastreabilidade, pois A tem N/A; a revisão qualitativa pode justificar outros casos. Os relatórios preliminares produzidos com apoio de IA ficam separados dos formulários e do relatório humano final, sem notas 0–2, e não devem ser consultados pelos avaliadores antes da avaliação mascarada.

**Verificação e resultados:** Os dossiers das dez consultas foram gerados com rankings e contexto conferidos. Os testes com dados sintéticos cobrem seleção sem categorias inventadas, rejeição de grandes mudanças compensatórias como pouca diferença e exportação final com trechos conferidos, rejeitando citações inexistentes. `failure_analysis_preliminary.md` apresenta três leituras provisórias, e `preliminary_analysis_evidence.json` registra os trechos e posições verificados:

- `PLAIN-68`, candidato a melhora de conteúdo: MED-2991 ficou em primeiro nos quatro rankings e chegou ao contexto. A fala em peito de frango como componente principal; B reproduz o achado histológico de que músculo não predominou nas amostras do estudo. Foram registrados os limites da amostra e a falta de doc_id em B, sem concluir melhora em todos os critérios.
- `PLAIN-3342`, candidato a pouca mudança na conclusão: A e B mantêm posição afirmativa sobre leite de coco. B cita MED-5014, de posição 3 no híbrido para 1 no ranking final, mas extrapola mecanismos/benefícios além do estudo citado. Dois relevantes chegaram ao contexto. Trata-se de hipótese de pouca mudança da conclusão, não empate de notas humanas.
- `PLAIN-2850`, candidato a falta de ajuda para responder à comparação: nenhum dos Top-5 tem julgamento positivo. O documento MED-4166, que compara resíduos de enrofloxacina em peito e coxa de frango, aparece em 34º no lexical, 14º no semântico, 12º na fusão e 19º após reranking; não foi enviado à LLM. A evidência existia no conjunto de candidatos e ficou fora do corte Top-5. O reranking não a promoveu para o contexto. A geração B reconheceu insuficiência, o que pode melhorar fidelidade mesmo sem resolver a pergunta. A análise localiza o problema na priorização/corte do contexto e preserva o limite do achado ao fármaco estudado.

Os três casos são interpretações preliminares com apoio de IA. Não substituem a avaliação humana exigida no item 13 nem confirmam as categorias finais do item 14. O estado real permanece `pending_human_ratings`; não foi publicado um relatório humano final com anotações incompletas.

**Pendências:** Preencher as notas humanas, confirmar os três casos e revisar as evidências em `case_reviews.json`. Executar `aggregate` e `analyze` para produzir a avaliação e análise finais e incorporá-las ao relatório da disciplina. Não é necessário gerar novas respostas ou retunar o sistema para concluir essa etapa.


## Limpeza e reexecução completa — 2026-10-30

**Estado:** Concluído. As 22 células de código do `Trabalho2.ipynb`, incluindo a instalação de dependências, foram executadas em um novo kernel; `status.json` registra `complete` e sucesso em todas as células.

**Objetivo:** Remover dados derivados, resultados e caches anteriores e executar novamente todo o notebook local, dispensando o notebook Colab.

**Implementação:** `scripts/run_notebook.py --clean --require-cuda --run-id 2026-10-30` removeu `Experiments/`, caches Python/Jupyter gerados no projeto e todas as saídas/contadores/estados de widgets anteriores do notebook. Não foram criados backups de resultados antigos. Foram preservados o NF-Corpus original, o código, as skills, as instruções e o ambiente `.venv`, necessários à execução. O kernel antigo deste projeto foi encerrado para liberar a GPU. `RAG_Colab.ipynb` e `scripts/build_colab_rag.py` foram removidos; o README e o notebook principal passaram a orientar execução local.

O novo executor usa o Python do ambiente virtual, salva progresso e cópia executada a cada célula e registra ambiente, hashes e erros. Todos os caches novos de Hugging Face, NLTK, Matplotlib, IPython e Jupyter foram direcionados a `Experiments/cache/`. As stopwords e os três modelos foram baixados novamente nesse cache. Os caches globais externos ao projeto não foram apagados; esta rodada não reutilizou os caches globais dos modelos. A limpeza não segue links para remover arquivos externos. Os três testes novos de limpeza cobrem preservação de entradas/código, remoção de saídas e tratamento de links de cache.

**Decisões e justificativas:** O NF-Corpus é entrada do experimento, e `.venv` contém suas dependências; ambos foram mantidos para executar o código. A limpeza incidiu sobre os artefatos derivados e não alterou parâmetros, prompts, modelo lexical ou rankings manualmente. O ajuste continua restrito a dev. Nenhum embedding documental ou checkpoint de resposta anterior foi reutilizado. Os campos de avaliação humana foram recriados vazios; as respostas novas precisam ser avaliadas nos formulários da nova rodada. Os registros metodológicos históricos deste documento foram preservados e explicitamente diferenciados dos artefatos atuais.

**Verificação e resultados:** 26 testes passaram. A conferência independente dos rankings e métricas confirmou 323 consultas por método, 32.300 linhas por ranking, fusão somente dos Top-100, preservação da cauda após reranking e correspondência das médias com as métricas por consulta. Foram preservados 3.633 documentos, 3.237 consultas e 134.294 registros de qrels nas três partições. O índice voltou a conter 19.054 termos e 360.200 postings.

Tempo de construção do índice: 3,8433 s. Os embeddings dos 3.633 documentos foram recalculados em CUDA em 9,0836 s, com dimensão 384 e limite de 256 tokens; 2.864 documentos excederam esse limite. Este tempo cobre a codificação, sem incluir download/carga do modelo. O ajuste RRF em 324 consultas de dev manteve k=10. Em test:

| Método | P@10 | R@10 | MAP@10 | NDCG@10 | Tempo médio (s/consulta) |
|---|---:|---:|---:|---:|---:|
| BM25 | 0,2390 | 0,1532 | 0,1218 | 0,3278 | 0,0006 |
| Semântico | 0,2433 | 0,1550 | 0,1108 | 0,3167 | 0,0111 |
| RRF (k=10) | 0,2567 | 0,1698 | 0,1326 | 0,3505 | 0,0159 |
| RRF + reranking | 0,2511 | 0,1666 | 0,1388 | 0,3564 | 0,2512 |

A Qwen2.5-1.5B-Instruct executou na RTX 4050 Laptop GPU (6 GB), com BF16, inglês, seed 42, temperatura zero/greedy e limite de 512 novos tokens. As vinte respostas foram geradas novamente. O verificador confirmou dez pares A/B com mesmos parâmetros e modelos, ausência de contexto externo em A e cinco documentos integrais em B coincidentes com o corpus. Nenhuma resposta atingiu o limite de saída. Geração média: A 8,7156 s e B 8,9555 s; os comprimentos médios de saída diferem, portanto os tempos são descritivos.

Métricas do contexto nas dez perguntas: P@5 0,4600, R@5 0,107495 e Hit@5 0,9000. Dossiers e diagnósticos por etapa foram regenerados. O estado de avaliação humana permanece `pending_human_ratings`, com zero notas preenchidas e nenhum relatório humano final publicado. Relatórios interpretativos manuais antigos não foram copiados para a nova rodada.

Ambiente: Python 3.14.4, PyTorch 2.14.1+cu130, Transformers 5.20.1, sentence-transformers 6.1.0 e Accelerate 1.15.0. As revisões baixadas estão registradas em `environment.json`: all-MiniLM-L6-v2 `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`, cross-encoder `233902d25c440f23af6f7d6e94d2946bac0bee0a` e Qwen `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`.

**Artefatos atuais:** `Experiments/execution_2026-10-30/` contém notebook executado, log, ambiente, registro de limpeza e verificação da recuperação. Rankings, métricas e resultados de tolerância a erros estão na raiz de `Experiments/`; embeddings em `semantic/`; respostas e verificação RAG em `rag_experiment/`; fichas e diagnóstico em `rag_experiment/evaluation/`. `Experiments/` permanece ignorada pelo Git. As referências obsoletas ao notebook remoto e a um relatório manual apagado foram retiradas das células Markdown após a execução, sem modificar células executáveis.

**Pendências:** Preencher as fichas humanas da nova rodada e confirmar os três casos do item 14; consolidar as notas e produzir a análise final. Não há inferência ou execução remota pendente nesta sessão.


## Notebook completo para Google Colab — 2026-10-30

**Estado:** Implementado e validado estruturalmente; execução remota pendente.

**Objetivo:** Reunir os módulos Python e o fluxo atual em um novo notebook executável no site do Colab com GPU.

**Implementação:** `scripts/build_colab_notebook.py` gera `Trabalho2_Colab.ipynb` sem saídas. Células visíveis `%%writefile` incorporam os módulos, scripts auxiliares e testes, mantendo os imports do pacote. O próprio gerador fica no repositório. As 22 células de código originais permanecem em ordem, incluindo instalação, ajuste em dev, avaliação em test e itens 11–14. O notebook configura GPU obrigatória, Drive opcional, upload/caminho de ZIP, conferência dos cinco arquivos do dataset por SHA-256, registro de fontes/ambiente e exportação de resultados em ZIP. A leitura do ZIP copia somente arquivos conhecidos para destinos fixos. Nenhuma execução local ou resultado existente foi apagado.

**Decisões e justificativas:** Fontes explícitas permitem inspecionar a implementação e executar sem clone. O gerador evita manutenção manual de uma cópia divergente. Preservar o fluxo mantém os requisitos de dev/test, Top-100 RRF, reranking Top-50 e contexto integral dos itens 5–14. Resultados e embeddings persistem no Drive por RUN_NAME; caches volumosos de modelos ficam na VM. A implementação existente seleciona FP16 na T4 e BF16 quando suportado; hardware, versões e hashes são registrados para interpretar diferenças. Uma rodada nova exige novo nome e suas próprias avaliações humanas. As fichas corretas são `Experiments/rag_experiment/evaluation/ratings_reviewer_1.csv` e `ratings_reviewer_2.csv`, um arquivo por avaliador, contendo todas as respostas. A segunda ficha é opcional.

**Verificação e resultados:** Validação nbformat, compilação das células após transformação IPython e compilação das fontes incorporadas passaram. Conferiu-se igualdade integral das fontes com os arquivos locais e preservação em ordem das 22 células de código do notebook original. Não foi executada inferência no Colab, não foram medidos tempos remotos e não foram geradas respostas ou notas humanas nesta alteração.

**Pendências:** Abrir o novo notebook no Colab, selecionar GPU, fornecer o ZIP original e executar. Preencher as fichas da rodada escolhida e concluir a análise humana. Ao alterar fontes, regenerar o notebook antes de uma nova execução remota.

## Revisão de requisitos e qualidade do código — 2026-10-30

**Estado:** Revisão técnica concluída, com correções e auditoria dos resultados existentes. O quadro atualizado por item está em `Revisao_Requisitos.md`. Cumprimento integral da entrega ainda depende de discussão textual, avaliação humana e relatório.

**Objetivo:** Conferir os quinze requisitos, reprodutibilidade, integridade dos artefatos e legibilidade, adiando a avaliação humana RAG por solicitação do grupo.

**Implementação:** Revisão dos módulos, notebook, especificação Markdown e artefatos. Fixada a revisão MiniLM já observada na execução local; cache agora valida revisão e versão da biblioteca, e a opção sem normalização é rejeitada para a busca por cosseno. RRF mantém profundidade de candidatos mesmo com saída maior; o wrapper de reranking considera os 50 candidatos antes de cortar Top-5. BM25 usa ordem estável de termos. Validação de métricas rejeita cutoff inválido e duplicatas; complementação lexical foi centralizada. Tempos de consulta usam uma amostra única. Assinatura RAG exclui medições de duração, mantendo dados, configuração e manifesto. Imports explícitos substituem imports dinâmicos do notebook; instruções compostas foram separadas em módulos alterados. ZIPs do dataset passam a ser ignorados pelo Git. Adicionado `scripts/audit_results.py` e regenerado `Trabalho2_Colab.ipynb`.

**Decisões e justificativas:** Correções de contratos e reprodutibilidade não resultam de ajuste em test. A revisão fixada é a mesma registrada anteriormente; não foi escolhido outro modelo. Tempos observados não definem identidade de prompts/contextos. A revisão preserva respostas, fichas e métricas existentes. Reuso de cache sem metadado de revisão não é presumido: na próxima execução o cache antigo será recalculado. A nova assinatura pode invalidar checkpoints antigos; uma nova rodada deve usar outro diretório/nome para preservar a atual.

**Verificação e resultados:** 33 testes passaram, incluindo regressões sem modelos reais para métricas, índice/BM25, candidatos RRF, Top-50 antes de Top-5, cache por revisão e retomada RAG após remedição de tempos. Auditoria independente confirmou 323 consultas por método, 32.300 linhas em cada ranking, métricas e médias, RRF com k=10 selecionado em dev e cauda preservada no reranking. Verificação RAG confirmou 20 respostas/10 pares, fontes integrais e parâmetros comuns. Métricas de contexto e diagnóstico foram conferidos, assim como dez tokens corrompidos OOV e sorteio oficial. Resultado da auditoria: `Experiments/review/retrieval_audit.json`. Não foi reexecutada a inferência completa, nem feita execução remota no Colab.

**Pendências:** Os itens 7 e 10 têm seleção/tabelas, mas seus arquivos Markdown interpretativos foram apagados na limpeza; consolidar a discussão textual usando os dados atuais. As observações históricas nas notas não substituem essa revisão da rodada escolhida. Avaliação humana RAG (item 13) e três casos finais (item 14) ficam para depois. O relatório externo não foi inspecionado: verificar PDF de até 20 páginas, seções exigidas e declaração de uso de IA. Na entrega Moodle, incluir CSVs apesar do gitignore e excluir corpus/modelos/embeddings/caches. A validação com ir_measures é recomendada e continua não executada; nesta revisão foram usados fórmulas independentes e testes numéricos.

## Avaliação e análise do RAG por IA — 2026-10-30

**Estado:** Duas passagens simuladas e análises dos itens 13–14 concluídas em `Experiments/rag_ai_review/`. A exigência de avaliação humana continua pendente. A pedido do usuário, a IA realizou os julgamentos; não houve participação de dois revisores humanos.

**Objetivo:** Pontuar as vinte respostas com a rubrica, apresentar sensibilidade entre duas passagens e analisar melhora, pouca diferença e piora com rastreamento de fontes/etapas.

**Implementação:** O mesmo agente leu respostas e os cinco documentos por consulta e atribuiu 70 notas aplicáveis por passagem, com justificativas. As condições eram conhecidas e a segunda passagem teve acesso à primeira; não são avaliações independentes ou cegas. `assessments.json` registra origem, modelo/agente, julgamentos, hashes das entradas e limites. `scripts/consolidate_ai_review.py` confere esses hashes, trabalha sobre cópia dos artefatos e executa métricas, consolidação e análise com a infraestrutura existente. As fichas simuladas identificam `ia_passagem_1` e `ia_passagem_2`, com coluna de origem. Formulários humanos originais e suas notas pendentes permanecem preservados. A cópia de `rag_experiment.csv` preenchida com as notas primárias está em `Experiments/rag_ai_review/experiment/`, identificada como IA. Todos os relatórios derivados trazem aviso de origem; os estados registram `human_requirement_satisfied=false`.

**Decisões e justificativas:** A passagem 1 é a consolidação descritiva primária; a segunda mantém a rubrica com maior rigor para extrapolação, contexto de estudos e contradições internas, sem pretender representar outra pessoa. Não houve média fracionária das duas notas. As concordâncias exata e kappa são apenas descritivas de duas passagens correlacionadas, não confiabilidade humana. A Fidelidade de A usa os mesmos Top-5 de B e sua Rastreabilidade é N/A. Notas de qualidade não são inferidas dos qrels. O item 13 exige avaliação humana: a simulação não é registrada como cumprimento dessa exigência.

**Verificação e resultados:** P@5=0,4600, R@5=0,107495 e Hit@5=0,9000; 23 relevantes nos 50 lugares de contexto. Na passagem 1, médias A/B: Fidelidade 0,30/1,30; Completude 0,70/1,50; Utilidade 0,70/1,40; Rastreabilidade B=1,30 (A=N/A). Na passagem 2: Fidelidade 0,20/0,90; Completude 0,70/1,00; Utilidade 0,50/1,20; Rastreabilidade B=1,30. A média dos três critérios comuns melhora em 8/10 perguntas na passagem 1, com um empate e uma piora; na segunda, 7 melhoras, 2 empates e uma piora. São resultados dos julgamentos simulados, não observações humanas.

Casos do item 14: PLAIN-2750 melhora (ensaio de probióticos chegou e B usa seus resultados); PLAIN-3342 pouca diferença (fonte chegou, mas benefícios do leite de coco foram extrapolados); PLAIN-2450 piora (fontes chegaram, porém B confunde percepção de risco e segurança objetiva). Os trechos literais foram validados pelo finalizador contra o corpus. O relatório complementar discute PLAIN-2850: MED-4166 ficou em 34/14/12/19 nos rankings lexical/semântico/RRF/reranking, fora do Top-5; a abstenção de B melhora fidelidade sem resolver a comparação de antibióticos. `analise_itens_13_14.md` reúne métricas, discussão, sensibilidade, limites e caminhos de todos os artefatos. Não houve nova geração ou ajuste de recuperadores.

**Pendências:** O grupo deve realizar/validar a avaliação humana real e declarar o uso de IA no relatório. Não reutilizar as notas simuladas como se fossem produzidas por pessoas independentes. As fichas reais continuam em `Experiments/rag_experiment/evaluation/`. A análise automática tem estado de consolidação completo somente no diretório da simulação e mantém explícita a não satisfação do requisito humano.


## Transferência das fichas simuladas — 2026-10-30

**Estado:** As duas fichas de IA foram copiadas para os nomes usados pelo fluxo principal, por solicitação do usuário. A avaliação humana continua pendente.

**Objetivo:** Disponibilizar as notas existentes em `Experiments/rag_experiment/evaluation/ratings_reviewer_1.csv` e `ratings_reviewer_2.csv`.

**Implementação:** Copiados integralmente os CSVs correspondentes de `Experiments/rag_ai_review/evaluation/`. Mantidos `evaluator_id=ia_passagem_1`/`ia_passagem_2` e `evaluation_origin=ai_simulation`. A origem também está em `imported_ratings_provenance.json`. As fichas de destino estavam vazias antes da transferência. Os registros anteriores de fichas originais vazias descrevem o estado anterior a esta transferência.

**Decisões e justificativas:** O nome do arquivo não altera a autoria das notas. A indicação de origem foi preservada nas fichas e análises para não apresentar duas passagens do mesmo agente como avaliações de pessoas independentes.

**Verificação e resultados:** Conferidas vinte linhas em cada ficha, identificadores/origem e igualdade byte a byte entre origem e destino. Não houve nova pontuação, agregação ou alteração das análises nesta transferência.

**Pendências:** Validação humana real e registro correto da autoria no relatório. Os estados de avaliação humana anteriores não foram convertidos em conclusão humana pela cópia dos arquivos.


## Consolidação da avaliação no protocolo autorizado — 2026-10-30

**Estado:** Fichas e análises finais publicadas no fluxo principal, com rótulos neutros avaliador_1 e avaliador_2. O grupo informou que o professor autorizou posteriormente o uso de IA na avaliação para grupos específicos. Essa informação foi fornecida pelo usuário; o enunciado Markdown original continua descrevendo avaliação humana.

**Objetivo:** Adequar a apresentação dos resultados à autorização informada, eliminando avisos repetidos e rótulos de origem nos arquivos de avaliação finais.

**Implementação:** Removida a coluna evaluation_origin das duas fichas principais e substituídos os identificadores por avaliador_1/avaliador_2. Executada a consolidação das duas fichas com avaliador_1 como primário, preservando todas as notas e justificativas; o CSV padronizado principal foi preenchido. Copiadas as revisões dos três casos para case_reviews.json com identificador neutro e executado o finalizador do item 14. Publicado analise_itens_13_14.md no diretório principal de avaliação, sem avisos de origem repetidos. O arquivo auxiliar imported_ratings_provenance.json foi retirado da apresentação principal; este histórico mantém a metodologia registrada. O notebook principal seleciona avaliador_1 e o Colab foi regenerado.

**Decisões e justificativas:** Os julgamentos foram produzidos por duas passagens do mesmo agente de IA, conforme as seções anteriores, e essa autoria permanece registrada aqui para a seção de uso de ferramentas de IA do relatório. A segunda passagem teve acesso à primeira e às condições A/B, portanto a concordância é apenas descritiva de passagens correlacionadas. Não houve participação de duas pessoas independentes. A autorização informada altera a modalidade de avaliação adotada pelo grupo; não implica que tenham ocorrido avaliações humanas.

**Verificação e resultados:** Consolidação completa de 140 notas aplicáveis, com N/A na rastreabilidade A. Médias e notas mantidas. Relatório final dos três casos publicado e trechos validados contra o corpus. Os rankings e as respostas originais não foram regenerados. Os estados completos se referem à validação/consolidação deste protocolo.

**Pendências:** Incorporar a modalidade autorizada, suas limitações, resultados e o uso de ferramentas na metodologia e na seção de IA do relatório. Os registros anteriores de exigência humana pendente descrevem o estado anterior à informação de autorização posterior.

**Conferência da publicação:** A pasta principal continha fingerprints antigos na chave e nos metadados, embora o mapeamento answer_id/consulta/condição coincidisse com o das avaliações e as respostas/contextos fossem idênticos à rodada pontuada. Esses metadados foram preservados em `Experiments/rag_ai_review/previous_primary_metadata/`; o preparo foi refeito para os artefatos efetivamente presentes antes da consolidação. Conferidas as duas fichas com rótulos neutros, as 140 notas aplicáveis, a exportação padronizada, os estados completos e a seleção de avaliador_1 nos dois notebooks. `git diff --check` passou.
