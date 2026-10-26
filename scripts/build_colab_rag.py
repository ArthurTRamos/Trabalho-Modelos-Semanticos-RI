"""Gera notebook Colab autossuficiente a partir dos contextos já recuperados.

Inclui snapshot de código e documentos Top-5, dispensando acesso do runtime aos
arquivos locais do VS Code. Não inclui respostas nem pesos de modelos.
"""
import argparse
import base64
import hashlib
import io
import json
from pathlib import Path
import zipfile


ROOT = Path(__file__).resolve().parents[1]


def build_notebook(contexts_path, output_path, revision='989aa7980e4cf806f80c7fef2b1adb7bc71aa306'):
    contexts = Path(contexts_path).read_bytes()
    bundle = json.loads(contexts)
    if len(bundle['cases']) != 10:
        raise ValueError('Prepare os dez contextos antes de gerar o notebook.')
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_DEFLATED) as file:
        file.writestr('ri_project/__init__.py', '# Snapshot mínimo para geração independente.\n')
        for name in ('rag.py', 'rag_experiment.py'):
            file.write(ROOT / 'ri_project' / name, f'ri_project/{name}')
        file.writestr('Experiments/rag_experiment/contexts.json', contexts)
    payload = archive.getvalue()
    digest = hashlib.sha256(payload).hexdigest()
    encoded = base64.b64encode(payload).decode('ascii')
    cells = []

    def cell(kind, source):
        entry = {'cell_type': kind, 'id': f'colab-rag-{len(cells)}',
                 'metadata': {}, 'source': source.splitlines(keepends=True)}
        if kind == 'code':
            entry.update(execution_count=None, outputs=[])
        cells.append(entry)

    cell('markdown', '''# Item 12 — Experimento sem RAG × com RAG no Colab

No VS Code, selecione **Select Kernel → Colab** e um servidor com GPU; execute todas as células. A sessão do chat não controla a seleção/autenticação da extensão.

Este notebook inclui os dez contextos recuperados no computador local e um snapshot do código. O runtime remoto não precisa enxergar o repositório nem o dataset local. A recuperação foi feita por BM25 + semântica → RRF ajustado em dev → reranking Top-50 → Top-5 integral. O manifesto incluído registra essa preparação.

Serão geradas 20 respostas em inglês: A sem documentos, B com documentos, mesma Qwen2.5-1.5B-Instruct, parâmetros, seed 42 e limite de 512 novos tokens. Cada chamada é independente, sem histórico ou cache de conversa compartilhado. A geração usa greedy; isso favorece repetibilidade, mas dispositivos/versões diferentes podem produzir respostas diferentes.

Os resultados abaixo só existem após a execução. Avaliação humana e análise dos itens 13–14 ficam pendentes. O Colab pode interromper o runtime; salve os resultados antes de encerrá-lo.
''')
    cell('code', '''import torch
if not torch.cuda.is_available():
    raise RuntimeError('Selecione um runtime Colab com GPU antes de continuar.')
print('GPU:', torch.cuda.get_device_name(0))
''')
    cell('code', '%pip install -q "transformers==4.57.6" "accelerate==1.15.0"\n')
    cell('code', f'''import base64, hashlib, io, os, sys, zipfile
from pathlib import Path

payload = base64.b64decode({encoded!r})
assert hashlib.sha256(payload).hexdigest() == {digest!r}
project = Path('/content/ri_rag_item12')
project.mkdir(parents=True, exist_ok=True)
with zipfile.ZipFile(io.BytesIO(payload)) as archive:
    archive.extractall(project)
os.chdir(project)
sys.path.insert(0, str(project))
print('Snapshot verificado e extraído em', project)
''')
    cell('code', f'''import json
from ri_project.rag import RAGConfig, LocalCausalGenerator
from ri_project.rag_experiment import run_paired_experiment

bundle = json.loads(Path('Experiments/rag_experiment/contexts.json').read_text())
print('Consultas oficiais:', bundle['query_ids'])
config = RAGConfig(revision={revision!r}, temperature=0.0, max_new_tokens=512, seed=42)
generator = LocalCausalGenerator(config, device='cuda')
print('Manifesto:', generator.manifest())
output_dir = Path('Experiments/rag_experiment_gpu')
answers = run_paired_experiment(bundle, generator, output_dir)
''')
    cell('code', '''from IPython.display import Markdown, display
display(Markdown((output_dir / 'comparison.md').read_text()))
print(json.loads((output_dir / 'run_status.json').read_text()))
''')
    cell('code', '''import shutil
from IPython.display import FileLink, display
# Inclui contextos originais junto das respostas para auditoria local.
shutil.copyfile('Experiments/rag_experiment/contexts.json', output_dir / 'contexts.json')
archive_path = shutil.make_archive('rag_item12_resultados_gpu', 'zip', output_dir)
display(FileLink(Path(archive_path).name))
print('No VS Code, também pode baixar o ZIP pelo explorador Colab → Contents → Download.')
''')
    notebook = {'cells': cells, 'metadata': {
        'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
        'language_info': {'name': 'python'}, 'accelerator': 'GPU',
        'snapshot_sha256': digest,
    }, 'nbformat': 4, 'nbformat_minor': 5}
    Path(output_path).write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + '\n')
    print(f'Notebook gerado: {output_path}; payload {len(payload):,} bytes; SHA256 {digest}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--contexts', default=str(ROOT / 'Experiments/rag_experiment/contexts.json'))
    parser.add_argument('--output', default=str(ROOT / 'RAG_Colab.ipynb'))
    parser.add_argument('--revision', default='989aa7980e4cf806f80c7fef2b1adb7bc71aa306')
    args = parser.parse_args()
    build_notebook(args.contexts, args.output, args.revision)
