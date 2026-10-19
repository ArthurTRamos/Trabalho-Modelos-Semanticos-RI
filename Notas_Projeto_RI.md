# Trabalho Prático 2 — Recuperação da Informação

Notas de implementação e decisões para apoiar a redação do relatório.

## Identificação do projeto

| Campo | Valor |
|---|---|
| Projeto | `Trabalho-Modelos-Semanticos-RI` |
| Notebook | `Trabalho2.ipynb` |
| Dataset | `nfcorpus/nfcorpus/` |
| Documento de requisitos | `Trabalho_Pratico_2_RI_2026.pdf` |

## Estado geral

O trabalho prevê BM25 (lexical), sentence-BERT (semântica), reranking e outros componentes especificados no PDF. Estão implementadas a leitura do dataset, o pré-processamento lexical, o índice invertido, a busca BM25, o experimento lexical de tolerância a erros, a recuperação semântica e a avaliação lexical × semântica em test. Busca híbrida, reranking e RAG ainda estão pendentes.

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

**Estado:** Não implementada.

**Objetivo:** A definir conforme o enunciado e o protocolo experimental.

**Implementação:** Ainda não há método de reranking registrado.

**Decisões e justificativas:** Pendentes: método/modelo de reranking, conjunto de candidatos recebido e configuração.

**Verificação e resultados:** Nenhum experimento executado.

**Pendências:** Consultar o PDF, definir o pipeline e implementar o reranking.

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

**Pendências:** Avaliar fusão híbrida, reranking e demais etapas do pipeline segundo os itens subsequentes do PDF. A escolha de modelo semântico foi fixada previamente por solicitação do projeto; caso o grupo queira comparar modelos alternativos, essa seleção deve ser feita exclusivamente em `dev`, sem alterar esta configuração com base no `test`.

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
