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

O trabalho prevê BM25 (lexical), sentence-BERT (semântica), reranking e outros componentes especificados no PDF. Até o momento foram implementadas a leitura do dataset, o pré-processamento lexical, o índice invertido, a busca BM25 e o experimento lexical de tolerância a erros. Recuperação semântica, busca híbrida, reranking e as demais avaliações ainda estão pendentes.

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

**Verificação e resultados:** A execução usada para o experimento carregou 3.633 documentos, 3.237 consultas e 134.294 linhas de qrels somadas das três partições. As colunas carregadas são `query-id`, `corpus-id` e `score`.

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

**Verificação e resultados:** Na execução de 2026-10-17: vocabulário lexical com 19.054 termos; 360.200 postings; comprimento médio processado de 165,82 tokens; tempo de construção medido em 4,5697 s. A medição começa depois da leitura do dataset e cobre tokenização, coleta do vocabulário pré-stemming, criação das posting lists e comprimentos. São valores de uma execução, sujeitos ao ambiente.

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

**Estado:** Implementada conforme o item 4 do PDF; experimento final executado em 2026-10-17 com dez consultas de `test`.

**Objetivo:** Corrigir termos desconhecidos com k-gramas para geração de candidatos, Jaccard para filtragem e distância de edição para seleção; medir a perda e recuperação de NDCG@10 no sistema lexical.

**Implementação:**

- `raw_vocabulary` é coletado dos tokens minúsculos do título e texto dos documentos antes de stemming/lematização, incluindo tokens que depois são removidos como stopwords. Este é o vocabulário usado para reconhecer termos conhecidos e para correção.
- `character_kgrams(term, k=3)` produz trigramas de caracteres com padding de início/fim; `kgram_index` associa cada trigrama aos termos do vocabulário e é criado antes do experimento.
- Para um termo OOV, `best_spelling_candidate` reúne termos via índice de k-gramas, filtra por coeficiente de Jaccard mínimo de 0,30 e seleciona por menor distância de Levenshtein (limite 2), depois maior Jaccard e ordem lexical determinística.
- `correct_query` mantém termos conhecidos e substitui termos desconhecidos quando há candidato que passa os limiares.
- `make_oov_typo` introduz uma única inserção, remoção, substituição ou transposição. Só aceita corrupções ausentes do vocabulário pré-stemming e que o corretor mapeia de volta ao termo original.
- `run_typo_experiment` exige exatamente 10 consultas quando `split='test'`, seleciona consultas com NDCG@10 original maior que zero e distribui os tipos de erro de forma cíclica. O experimento compara consulta original, consulta corrompida e consulta corrigida usando BM25 e NDCG@10 com ganho linear e IDCG sobre todos os julgamentos da consulta.
- Para cada consulta, quando `NDCG10_original - NDCG10_erro > 0`, calcula `(NDCG10_corrigida - NDCG10_erro) / (NDCG10_original - NDCG10_erro)`. Nos outros casos registra “erro não reduziu o desempenho”. O resultado é salvo em `typo_experiment.csv` com as colunas mínimas exigidas pelo PDF e detalhes adicionais dos tokens/correções.

**Decisões e justificativas:** Trigramas com marcadores de fronteira ajudam a distinguir prefixos/sufixos e são uma escolha comum para erros curtos. Jaccard reduz candidatos antes do cálculo mais custoso de distância de edição. Levenshtein resolve a seleção pela menor quantidade de inserções, remoções ou substituições; transposição adjacente pode resultar em distância 2 nesta métrica. Os limiares 0,30 e 2 são valores iniciais fixados para esta execução; o código permite ajustá-los em `dev`, mas não foram calibrados nesta execução. A seleção de consultas de teste usa os qrels somente para cumprir o critério obrigatório de NDCG original positivo; nenhum parâmetro foi ajustado com `test`. As outras buscas continuam recebendo as consultas originais, sem correção.

**Verificação e resultados:** A execução carregou 3.633 documentos, 3.237 consultas, construiu vocabulário pré-stemming de 26.475 termos (índice de k-gramas em 0,1024 s) e gerou 10 linhas únicas em `typo_experiment.csv`. Foram usadas inserção (3), remoção (3), substituição (2) e troca de ordem (2). Os dez valores de NDCG@10 original foram positivos; todos os termos artificiais estavam ausentes do vocabulário pré-stemming e os dez termos corrigidos voltaram ao token original. Em 8 consultas o erro reduziu o desempenho e a fração média recuperada foi 1,0; em 2 consultas o erro não reduziu o desempenho conforme a regra do PDF. IDs dos dois casos sem perda: `PLAIN-1109` e `PLAIN-1151`. A execução usou `k1=1.2`, `b=0.75`, Jaccard mínimo 0,30 e distância máxima 2.

O NLTK tentou buscar stopwords, mas a rede do ambiente não validou o host. O recurso já estava instalado localmente, então a execução prosseguiu. O notebook foi ajustado para só tentar o download se as stopwords não estiverem disponíveis localmente.

**Pendências:** Avaliar sensibilidade dos limiares usando apenas `dev`, se desejado; manter o CSV entre os artefatos finais. Os valores medidos acima são uma execução registrada, não evidência de que a configuração seja ótima.

### Etapa 6 — Recuperação semântica com sentence-BERT

**Estado:** Não implementada.

**Objetivo:** A definir conforme o enunciado e o protocolo experimental.

**Implementação:** Ainda não há modelo nem código de recuperação semântica registrado.

**Decisões e justificativas:** Pendentes: modelo sentence-BERT, estratégia de codificação, normalização de embeddings e medida de similaridade.

**Verificação e resultados:** Nenhum experimento executado.

**Pendências:** Consultar o PDF, escolher e documentar a configuração e implementar a recuperação semântica.

### Etapa 7 — Reranking

**Estado:** Não implementada.

**Objetivo:** A definir conforme o enunciado e o protocolo experimental.

**Implementação:** Ainda não há método de reranking registrado.

**Decisões e justificativas:** Pendentes: método/modelo de reranking, conjunto de candidatos recebido e configuração.

**Verificação e resultados:** Nenhum experimento executado.

**Pendências:** Consultar o PDF, definir o pipeline e implementar o reranking.

### Etapa 8 — Avaliação experimental

**Estado:** Não implementada.

**Objetivo:** Comparar as técnicas sob protocolo e métricas especificados no PDF.

**Implementação:** Ainda não há protocolo de avaliação implementado.

**Decisões e justificativas:** Pendentes: métricas, partições, parâmetros e protocolo comparável entre as técnicas.

**Verificação e resultados:** Nenhuma avaliação executada; não há resultados experimentais para relatar.

**Pendências:** Conferir no PDF requisitos, estrutura do relatório, condições experimentais e forma de entrega. Selecionar configurações em `dev`, congelá-las e só então executar avaliação final em `test`.

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
- `measure_bm25_query_time(split, top_k, repetitions)`: tempo médio de busca por consulta numa partição de qrels.
- `kgram_index`, `character_kgrams`, `jaccard_coefficient`, `edit_distance`, `correct_query` e `run_typo_experiment`: geração, filtragem, seleção de correção e experimento OOV.
- `typo_experiment.csv`: resultados por consulta das condições original, com erro e corrigida.

## Registro de execução das notas

- `Notas_Projeto_RI.txt` foi convertido para este arquivo Markdown, mantendo o conteúdo técnico e reorganizando cada etapa com os campos **estado**, **objetivo**, **implementação**, **decisões e justificativas**, **verificação e resultados** e **pendências**.
- Em 2026-10-17, as células do notebook foram executadas para o experimento lexical de tolerância a erros; resultados e limitações estão registrados nas Etapas 1–5. Não foram executados os demais experimentos.
