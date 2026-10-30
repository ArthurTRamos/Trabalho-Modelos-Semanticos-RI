"""Executa Trabalho2 em kernel isolado; --clean remove artefatos anteriores.

A coleção NF-Corpus e o ambiente virtual são entradas preservadas. Todos os
caches desta execução são direcionados a Experiments/cache/.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import sys
import time
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    temporary.replace(path)


def clear_outputs(notebook):
    for cell in notebook['cells']:
        if cell['cell_type'] == 'code':
            cell['outputs'] = []
            cell['execution_count'] = None
            cell.get('metadata', {}).pop('execution', None)
    notebook.get('metadata', {}).pop('widgets', None)
    return notebook


def clean_generated_files():
    """Limpeza limitada ao projeto, sem seguir links para caches globais."""
    removed = []
    experiments = ROOT / 'Experiments'
    if experiments.is_symlink():
        experiments.unlink()
        removed.append('Experiments/ (link)')
    elif experiments.exists():
        shutil.rmtree(experiments)
        removed.append('Experiments/')
    excluded = {'.git', '.venv', 'nfcorpus', '.agents', '.codex', '.aws'}
    cache_names = {'__pycache__', '.pytest_cache', '.ipynb_checkpoints', '.cache'}
    for directory, dirs, _ in os.walk(ROOT, topdown=True, followlinks=False):
        for name in list(dirs):
            if name in excluded:
                dirs.remove(name)
            elif name in cache_names:
                path = Path(directory) / name
                if path.is_symlink():
                    path.unlink()
                else:
                    shutil.rmtree(path)
                removed.append(str(path.relative_to(ROOT)) + '/')
                dirs.remove(name)
    source = ROOT / 'Trabalho2.ipynb'
    notebook = clear_outputs(json.loads(source.read_text()))
    source.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + '\n')
    return removed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--clean', action='store_true')
    parser.add_argument('--require-cuda', action='store_true')
    parser.add_argument('--run-id', default=datetime.now(ZoneInfo('America/Sao_Paulo')).date().isoformat())
    args = parser.parse_args()
    if not args.run_id or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in args.run_id):
        parser.error('Use um identificador simples de execução, por exemplo 2026-10-30.')
    os.chdir(ROOT)
    removed = clean_generated_files() if args.clean else []
    output_dir = ROOT / 'Experiments' / f'execution_{args.run_id}'
    output_dir.mkdir(parents=True, exist_ok=True)
    cache = ROOT / 'Experiments' / 'cache'
    env_dirs = {
        'HF_HOME': cache / 'huggingface',
        'HF_HUB_CACHE': cache / 'huggingface' / 'hub',
        'NLTK_DATA': cache / 'nltk', 'MPLCONFIGDIR': cache / 'matplotlib',
        'IPYTHONDIR': cache / 'ipython',
        'JUPYTER_RUNTIME_DIR': cache / 'jupyter' / 'runtime',
    }
    for key, path in env_dirs.items():
        path.mkdir(parents=True, exist_ok=True)
        os.environ[key] = str(path)
    os.environ.pop('HF_HUB_OFFLINE', None)
    os.environ.pop('TRANSFORMERS_OFFLINE', None)
    os.environ.update(TOKENIZERS_PARALLELISM='false', OMP_NUM_THREADS='4',
                      MKL_NUM_THREADS='4', PIP_NO_CACHE_DIR='1',
                      PIP_DISABLE_PIP_VERSION_CHECK='1', PYTHONDONTWRITEBYTECODE='1')
    print('Preparando caches e ambiente de execução; '
          f'limpeza solicitada: {args.clean}.', flush=True)
    import nltk
    if not nltk.download('stopwords', download_dir=str(env_dirs['NLTK_DATA']), quiet=True,
                         raise_on_error=True):
        raise RuntimeError('Não foi possível baixar as stopwords no cache novo.')
    # A limpeza ocorre antes de importar PyTorch/Jupyter e criar o kernel.
    import torch
    import nbformat
    from nbclient import NotebookClient
    from jupyter_client import KernelManager

    if args.require_cuda and not torch.cuda.is_available():
        raise RuntimeError('CUDA indisponível; execute em um terminal com acesso à GPU.')
    start = datetime.now(timezone.utc).isoformat()
    environment = {
        'started_at_utc': start, 'run_id': args.run_id, 'python': sys.version,
        'python_executable': sys.executable,
        'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        'gpu_total_memory_bytes': (torch.cuda.get_device_properties(0).total_memory
                                   if torch.cuda.is_available() else None),
        'cuda_version': torch.version.cuda,
        'versions': {name: importlib.metadata.version(name) for name in (
            'torch', 'transformers', 'sentence-transformers', 'accelerate',
            'numpy', 'pandas', 'nltk', 'nbclient', 'nbformat')},
        'cache_directories': {k: str(v.relative_to(ROOT)) for k, v in env_dirs.items()},
        'dependency_install_cell_skipped': False,
        'package_sha256': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                            for p in sorted((ROOT / 'ri_project').glob('*.py'))},
    }
    write_json(output_dir / 'environment.json', environment)
    write_json(output_dir / 'cleanup.json', {
        'clean_requested': args.clean, 'removed': removed,
        'preserved_inputs': ['nfcorpus/', '.venv/', 'ri_project/', 'skills/',
                             'Trabalho_Pratico_2_RI_2026.md', 'Notas_Projeto_RI.md'],
        'old_results_backed_up': False,
    })
    source = ROOT / 'Trabalho2.ipynb'
    notebook = nbformat.read(source, as_version=4)
    clear_outputs(notebook)
    expected_sources = {c.id: c.source for c in notebook.cells}
    environment['notebook_source_sha256'] = hashlib.sha256(json.dumps(expected_sources, sort_keys=True).encode()).hexdigest()
    write_json(output_dir / 'environment.json', environment)
    status = {'status': 'running', 'started_at_utc': start, 'completed_code_cells': []}
    starts = {}
    log = (output_dir / 'execution.log').open('w', encoding='utf-8')

    def emit(message):
        print(message, flush=True)
        log.write(message + '\n')
        log.flush()

    def save():
        nbformat.write(notebook, output_dir / 'Trabalho2.executed.ipynb')
        # Não apaga alterações concorrentes de código feitas no editor.
        current = nbformat.read(source, as_version=4)
        unchanged = ({c.id: c.source for c in current.cells} == expected_sources)
        if unchanged:
            nbformat.write(notebook, source)
        else:
            status['source_modified_during_execution'] = True
        write_json(output_dir / 'status.json', status)

    def on_start(cell, cell_index, **_):
        if cell.cell_type == 'code':
            starts[cell_index] = time.perf_counter()
            status['current_cell_index'] = cell_index
            write_json(output_dir / 'status.json', status)
            emit(f'INÍCIO célula {cell_index}: {cell.source.splitlines()[0][:100]}')

    def on_done(cell, cell_index, execute_reply=None, **_):
        if cell.cell_type == 'code':
            elapsed = time.perf_counter() - starts[cell_index]
            success = (execute_reply or {}).get('content', {}).get('status') == 'ok'
            status['completed_code_cells'].append({'index': cell_index, 'id': cell.id,
                                                    'seconds': elapsed, 'success': success})
            save()
            emit(f'FIM célula {cell_index} ({elapsed:.2f}s; success={success})')
            for item in cell.outputs:
                if item.output_type == 'stream':
                    emit(item.get('text', '')[-2200:])
                elif item.output_type == 'error':
                    emit(item.get('ename', '') + ': ' + item.get('evalue', ''))

    manager = KernelManager(kernel_name='python3')
    manager.kernel_spec.argv = [sys.executable, '-m', 'ipykernel_launcher', '-f', '{connection_file}']
    client = NotebookClient(notebook, km=manager, timeout=3600,
                            resources={'metadata': {'path': str(ROOT)}},
                            on_cell_start=on_start, on_cell_executed=on_done)
    write_json(output_dir / 'status.json', status)
    try:
        client.execute()
        status['status'] = 'complete'
        emit('EXECUÇÃO COMPLETA')
    except BaseException as error:
        status.update(status='failed', error=repr(error))
        emit('FALHA: ' + repr(error))
        raise
    finally:
        status['finished_at_utc'] = datetime.now(timezone.utc).isoformat()
        save()
        if manager.has_kernel:
            manager.shutdown_kernel(now=True)
        environment['downloaded_snapshots'] = {
            p.parent.name: sorted(c.name for c in p.iterdir())
            for p in (cache / 'huggingface' / 'hub').glob('models--*/snapshots')}
        write_json(output_dir / 'environment.json', environment)
        log.close()


if __name__ == '__main__':
    main()
