# Revisão dos requisitos — 30/10/2026

Fonte primária: `Trabalho_Pratico_2_RI_2026.md`, itens numerados de **Requisitos** (a numeração do roadmap é diferente). Revisão de código, notebooks, testes e artefatos locais da execução de 30/10. A avaliação humana das respostas RAG fica para uma etapa posterior, conforme solicitado.

## Situação por requisito

| Item | Situação verificada | Evidências e limites |
|---|---|---|
| 1. Pré-processamento | Implementado | `preprocessing.py`: minúsculas, tokenização, stopwords inglesas, Porter. Título e texto compõem os documentos. Escolhas registradas nas notas; devem integrar o relatório. |
| 2. Índice invertido | Implementado e executado | `indexing.py` usa dicionários e contadores próprios, TF, DF via tamanho dos postings, comprimentos e média. 19.054 termos e 360.200 postings na rodada local; tempo de construção no notebook/log. BM25 pontua somente postings. |
| 3. Lexical | Implementado e executado | BM25 único em todos os experimentos, k1=1,2 e b=0,75. Justificativa frente a TF-IDF nas notas. Tempos em `metrics_summary.csv`. |
| 4. Erros | Implementado e executado | K-gramas/Jaccard/Levenshtein; dicionário antes do stemming. Dez consultas test elegíveis, quatro operações, tokens OOV conferidos. Fração recuperada e caso sem perda registrados. Ajuste separado em dev; demais buscas usam consultas originais. |
| 5. Semântica | Implementado e executado; reprodutibilidade reforçada | MiniLM, 384 dimensões, vetores normalizados/cosseno, embeddings persistidos. Limite 256 e 2.864 documentos acima dele registrados. Revisão dos pesos agora fixada e validada no cache. |
| 6. Métricas | Implementado e auditado | Mesmas 323 consultas; P10/R10/AP10/NDCG10 por consulta. AP/Recall dividem por todos os relevantes; NDCG linear e ideal dos qrels completos. Auditoria independente dos CSVs confirmou médias e MAP. |
| 7. Lexical × semântico | Parte quantitativa concluída; discussão final pendente | Seis casos e 60 linhas Top-5 em `comparison_cases_summary.csv`/`comparison_cases_top5.csv`, com textos, scores, graus e termos compartilhados. O antigo Markdown interpretativo foi apagado na limpeza. É necessário redigir/revisar as razões textuais na rodada escolhida; tabelas e interseções lexicais não comprovam por si só sinonímia ou mecanismos do modelo. |
| 8. Híbrido | Implementado e auditado | RRF, ranks iniciados em 1, contribuição só quando presente, Top-100 por método. Dev avalia 10/60/100 e seleciona k=10; test usa esse valor. Corrigido limite de candidatos para chamadas com saída maior que 100. |
| 9. Reranking | Implementado e auditado | Cross-encoder fixado, Top-50 reordenado, posições 51–100 preservadas. Dev antes de test. Tempos separados e truncamento no manifesto. Corrigida a chamada Top-5 para preservar os 50 candidatos do reranking. |
| 10. Análise do reranking | Casos e movimentos concluídos; discussão final pendente | Dois maiores ganhos, caso estável e piora em CSVs. As notas preservam observações históricas, mas o Markdown interpretativo foi apagado. Consolidar hipóteses com textos e posições da rodada escolhida no relatório. |
| 11. Pipeline RAG | Implementado e executado | Híbrido → reranking → cinco documentos integrais → Qwen. Prompt exige fontes e insuficiência de evidência. Não usa qrels para escolher contexto. Manifesto registra modelo/revisão, idioma, geração, hardware, precisão, tokens e custo de API zero. |
| 12. A/B | Implementado e auditado | Sorteio oficial conferido; 10 consultas/20 respostas. Mesma LLM e parâmetros, chamadas independentes, A sem coleção, B com Top-5 integral. Checkpoints agora desconsideram tempos voláteis da recuperação na assinatura. |
| 13. Avaliação RAG | Métricas concluídas; avaliação humana adiada | P5/R5/Hit5 recalculados e conferidos para os contextos reais. Fichas e rubrica existem; campos vazios continuam pendentes. A usa as mesmas fontes de B para avaliação de fidelidade; rastreabilidade A é N/A. |
| 14. Falhas ponta a ponta | Infraestrutura concluída; análise humana adiada | Diagnósticos conferidos com rankings e contexto. Três casos finais dependem de notas e revisão de evidências reais. Nenhum resultado humano foi produzido nesta revisão. |
| 15. CSVs padronizados | Estruturas presentes; notas humanas pendentes | Quatro rankings com 32.300 linhas cada, métricas por consulta, typo e `rag_experiment/rag_experiment.csv`. As notas aplicáveis deste último estão vazias até a avaliação humana. Os CSVs precisam ser incluídos no arquivo final entregue, mesmo estando ignorados pelo Git. |

## Correções e manutenção

- Modelo semântico fixado na revisão `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`, já observada na execução local. Cache verifica revisão e versão de Sentence Transformers. Configuração sem normalização é rejeitada porque o produto interno só corresponde ao cosseno com vetores normalizados.
- Limite de candidatos RRF independente do tamanho de saída. Reranking sempre considera a cabeça de 50 antes de cortar Top-5.
- BM25 acumula termos em ordem estável, evitando variação de soma decorrente de hashes do processo.
- Métricas rejeitam cutoff inválido e documentos duplicados. Complementação lexical reutiliza a função comum; medição de tempo usa uma única amostra por busca.
- Imports dinâmicos do notebook substituídos por imports explícitos. Instruções separadas por ponto e vírgula foram removidas de módulos alterados; funções modificadas receberam documentação de seus contratos. A organização por módulos foi mantida.
- ZIPs locais `nfcorpus*.zip` foram incluídos no gitignore. Não foram apagados.
- `scripts/audit_results.py` permite repetir a auditoria independente sem carregar modelos. O notebook Colab foi regenerado com todas as fontes atuais.

## Verificações e alcance

Foram executados 33 testes, incluindo novos contratos de postings/BM25, fórmulas das métricas, limites RRF/reranking, cache semântico e retomada RAG com tempos diferentes. Modelos nos testes novos são simulados: não há downloads nem inferência neural nessa suíte.

A auditoria dos resultados existentes confirmou as 323 consultas de cada método, 100 documentos únicos por ranking, graus dos qrels, quatro métricas e médias, RRF com k selecionado em dev, reordenação dos 50 candidatos e preservação da cauda. Também foram verificados os dez erros OOV, o sorteio oficial, os vinte registros RAG e a correspondência dos documentos integrais com o corpus. Os diagnósticos de contexto foram refeitos em diretório temporário.

Os resultados existentes continuam sendo os da execução anterior às correções desta revisão. Não houve nova rodada completa de inferência. Os casos-limite corrigidos de profundidade não ocorriam no fluxo original Top-100. A fixação de revisão e a ordem de soma não representam tuning em test. O cache antigo não contém a nova informação de revisão e será recalculado na próxima execução; a mudança na assinatura pode invalidar checkpoints antigos. Para uma nova rodada completa, use outra pasta/nome de execução e avalie as respostas dessa rodada. As respostas locais atuais continuam disponíveis para avaliação, sem necessidade de regenerá-las.

A execução remota no Colab permanece não verificada. A validação local do notebook cobre formato, sintaxe, correspondência das fontes e preservação do fluxo original.

## Entrega e trabalho restante

1. Redigir/revisar a discussão textual dos seis casos do item 7 e dos quatro casos do item 10, com as evidências da rodada escolhida.
2. Posteriormente, preencher as fichas RAG, consolidar as notas e revisar os três casos do item 14.
3. Preparar o relatório PDF de até 20 páginas (somente referências fora do limite), com todas as seções exigidas, inclusive **Uso de ferramentas de IA**. O relatório externo não foi inspecionado nesta revisão.
4. Montar o arquivo de entrega no Moodle com fontes, README e CSVs completos. Excluir dataset, pesos, embeddings, ambientes e caches. O ZIP de resultados do Colab é uma cópia para transferência e inclui embeddings; não é, sozinho, o pacote final de entrega.

A comparação com `ir_measures` é recomendada pelo enunciado, não obrigatória. Não foi feita; foram usados cálculos independentes e testes numéricos. Um segundo avaliador e o mascaramento são recomendações, já suportadas pelo código.
