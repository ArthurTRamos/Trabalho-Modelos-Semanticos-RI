"""Empacota fontes visíveis e fluxo local em um notebook independente para Colab."""
from pathlib import Path
import copy
import hashlib
import json
import textwrap

ROOT = Path(__file__).resolve().parents[1]


def build_notebook():
    import nbformat
    cells = []

    def md(source):
        cells.append(nbformat.v4.new_markdown_cell(textwrap.dedent(source).strip()))

    def code(source):
        cells.append(nbformat.v4.new_code_cell(textwrap.dedent(source).strip() + '\n'))

    md('''
    # Trabalho 2 — execução completa no Google Colab

    Este notebook contém as fontes Python e todas as etapas de `Trabalho2.ipynb`.
    Não é necessário clonar o repositório. Gerado por `scripts/build_colab_notebook.py`;
    regenere-o quando alterar os módulos ou o notebook principal.

    1. Abra este arquivo no site do Colab e selecione um ambiente com GPU (T4 é suficiente).
    2. Execute as células em ordem. Autorize o Drive para persistir resultados.
    3. Informe o caminho de um ZIP do NF-Corpus no Drive, ou faça upload quando solicitado.
       O ZIP deve conter `corpus.jsonl`, `queries.jsonl` e `qrels/{train,dev,test}.tsv`;
       são aceitas pastas externas como `nfcorpus/nfcorpus/`.
    4. Aguarde os experimentos. As fichas humanas ficam pendentes até seu preenchimento.

    O código usa a GPU do ambiente conectado. Na T4, a LLM seleciona FP16;
    em GPUs compatíveis pode selecionar BF16. Tempos e respostas podem diferir da rodada local.
    Use um novo `RUN_NAME` para uma rodada independente; mantenha-o para retomar a mesma rodada.
    Resultados locais não são importados nem sobrescritos por este notebook.
    ''')
    code('''
    from pathlib import Path
    import os
    import sys
    import re

    USE_DRIVE = True
    RUN_NAME = 'trabalho2_colab_01'
    # Exemplo: '/content/drive/MyDrive/nfcorpus.zip'. Vazio abre o upload.
    DATASET_ZIP = ''

    if any(name == 'ri_project' or name.startswith('ri_project.') for name in sys.modules):
        raise RuntimeError('Reinicie a sessão antes de executar novamente a preparação. '
                           'Para continuar avaliações, execute apenas as células finais.')
    if not re.fullmatch(r'[A-Za-z0-9_-]+', RUN_NAME):
        raise ValueError('RUN_NAME deve conter apenas letras, números, _ ou -.')
    WORKSPACE = Path('/content/trabalho2') / RUN_NAME
    WORKSPACE.mkdir(parents=True, exist_ok=True)
    if USE_DRIVE:
        from google.colab import drive
        drive.mount('/content/drive')
        RUN_DIR = Path('/content/drive/MyDrive/Trabalho2_RI') / RUN_NAME
    else:
        RUN_DIR = WORKSPACE / 'saved'
        print('Sem Drive: baixe os resultados antes de encerrar a sessão.')
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    results = RUN_DIR / 'Experiments'
    results.mkdir(exist_ok=True)
    link = WORKSPACE / 'Experiments'
    if link.is_symlink():
        if link.resolve() != results.resolve():
            raise RuntimeError('Destino de resultados mudou; use outro RUN_NAME.')
    elif link.exists():
        raise RuntimeError('Experiments já existe sem vínculo; use outro RUN_NAME.')
    else:
        link.symlink_to(results, target_is_directory=True)
    os.chdir(WORKSPACE)
    sys.path.insert(0, str(WORKSPACE))
    for directory in ('ri_project', 'scripts', 'tests'):
        Path(directory).mkdir(exist_ok=True)
    # Pesos grandes ficam no disco rápido e temporário da VM; resultados ficam no Drive.
    cache = WORKSPACE / 'cache'
    for variable, subdir in [('HF_HOME', 'huggingface'), ('HF_HUB_CACHE', 'huggingface/hub'),
                             ('NLTK_DATA', 'nltk'), ('MPLCONFIGDIR', 'matplotlib')]:
        target = cache / subdir
        target.mkdir(parents=True, exist_ok=True)
        os.environ[variable] = str(target)
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    print('Resultados persistidos em:', results)
    ''')
    source = nbformat.read(ROOT / 'Trabalho2.ipynb', as_version=4)
    md('## Dependências\nMantém as versões do projeto e o PyTorch disponibilizado pelo ambiente quando compatível.')
    cells.append(copy.deepcopy(source.cells[2]))
    code('''
    import torch
    import nltk
    if not torch.cuda.is_available():
        raise RuntimeError('Selecione GPU nas configurações do ambiente do Colab e reconecte.')
    if not nltk.download('stopwords', download_dir=os.environ['NLTK_DATA'], quiet=True):
        raise RuntimeError('Falha no download das stopwords do NLTK.')
    print('GPU:', torch.cuda.get_device_name(0))
    print('PyTorch:', torch.__version__, '| CUDA:', torch.version.cuda)
    ''')
    md('''
    ## Fontes incorporadas
    Cada célula `%%writefile` materializa um arquivo Python no ambiente temporário.
    As implementações são idênticas às do projeto, incluindo imports relativos.
    Os scripts e testes também estão disponíveis, mas não são executados automaticamente.
    ''')
    paths = sorted((ROOT / 'ri_project').glob('*.py'))
    paths += sorted(p for p in (ROOT / 'scripts').glob('*.py') if p.name != Path(__file__).name)
    paths += sorted((ROOT / 'tests').glob('*.py'))
    hashes = {}
    for path in paths:
        relative = path.relative_to(ROOT).as_posix()
        content = path.read_text(encoding='utf-8')
        hashes[relative] = hashlib.sha256(content.encode()).hexdigest()
        md(f'### `{relative}`')
        cells.append(nbformat.v4.new_code_cell(f'%%writefile {relative}\n{content}'))
    md('## Dataset e registro da execução')
    code('''
    import hashlib
    import json
    import shutil
    import zipfile
    from datetime import datetime, timezone
    from importlib.metadata import version

    required = ('corpus.jsonl', 'queries.jsonl', 'qrels/train.tsv', 'qrels/dev.tsv', 'qrels/test.tsv')
    data_dir = WORKSPACE / 'nfcorpus' / 'nfcorpus'
    if not all((data_dir / name).is_file() for name in required):
        if not DATASET_ZIP:
            from google.colab import files
            uploaded = files.upload()
            candidates = [name for name in uploaded if name.lower().endswith('.zip')]
            if len(candidates) != 1:
                raise ValueError('Envie exatamente um arquivo ZIP do NF-Corpus.')
            DATASET_ZIP = str(WORKSPACE / candidates[0])
            del uploaded
        with zipfile.ZipFile(DATASET_ZIP) as archive:
            names = set(archive.namelist())
            prefixes = [name[:-len('corpus.jsonl')] for name in names
                        if name.endswith('corpus.jsonl')
                        and all(name[:-len('corpus.jsonl')] + part in names for part in required)]
            if len(prefixes) != 1:
                raise ValueError('O ZIP precisa conter exatamente um NF-Corpus completo.')
            data_dir.mkdir(parents=True, exist_ok=True)
            # Copia apenas nomes conhecidos para destinos fixos, sem extrair caminhos do ZIP.
            for name in required:
                destination = data_dir / name
                destination.parent.mkdir(exist_ok=True)
                with archive.open(prefixes[0] + name) as src, destination.open('wb') as dst:
                    shutil.copyfileobj(src, dst)
    dataset_hashes = {name: hashlib.sha256((data_dir / name).read_bytes()).hexdigest()
                      for name in required}
    expected = EXPECTED_DATASET_HASHES
    if dataset_hashes != expected:
        raise ValueError('O dataset difere dos arquivos originais desta versão do projeto.')
    source_hashes = SOURCE_HASHES
    for name, digest in source_hashes.items():
        if hashlib.sha256(Path(name).read_bytes()).hexdigest() != digest:
            raise ValueError(f'Fonte alterada: {name}. Regenere o notebook para registrar a alteração.')
    manifest = {
        'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'gpu': torch.cuda.get_device_name(0), 'python': sys.version,
        'versions': {name: version(name) for name in
                     ('torch', 'transformers', 'sentence-transformers', 'accelerate', 'numpy', 'pandas')},
        'dataset_sha256': dataset_hashes, 'source_sha256': source_hashes,
        'run_name': RUN_NAME,
    }
    audit = results / 'colab_execution'
    audit.mkdir(exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    (audit / f'environment_{stamp}.json').write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    print('Dataset conferido; ambiente registrado em:', audit)
    '''.replace('EXPECTED_DATASET_HASHES', repr({name: hashlib.sha256((ROOT / 'nfcorpus/nfcorpus' / name).read_bytes()).hexdigest() for name in ('corpus.jsonl', 'queries.jsonl', 'qrels/train.tsv', 'qrels/dev.tsv', 'qrels/test.tsv')})).replace('SOURCE_HASHES', repr(hashes)))
    md('## Fluxo original\nAs células abaixo preservam o código e a ordem do notebook principal, após a instalação.')
    cells.extend(copy.deepcopy(source.cells[3:]))
    md('''
    ## Salvar e avaliar os resultados
    No Drive: `MyDrive/Trabalho2_RI/<RUN_NAME>/Experiments/`.
    Leia `rag_experiment/evaluation/human_review.md` e preencha
    `ratings_reviewer_1.csv` (segundo avaliador opcional: `ratings_reviewer_2.csv`).
    Cada ficha reúne as respostas de todas as consultas para um avaliador.
    Após preencher, reexecute as células de consolidação e análise; para dois avaliadores
    completos, configure `FINAL_REVIEWER`. Revise também `case_reviews.json`.
    Em uma nova sessão, execute preparação, dependências, fontes e dataset; depois execute
    apenas as seções dos itens 13–14 para consolidar avaliações já salvas, sem repetir a geração.

    A célula abaixo cria um ZIP dos resultados, incluindo embeddings e fichas, sem pesos de modelos.
    Para baixar no navegador, altere `DOWNLOAD_RESULTS` para `True`.
    Salve também uma cópia deste notebook com suas saídas pelo menu Arquivo do Colab.
    ''')
    code('''
    import zipfile
    DOWNLOAD_RESULTS = False
    archive_path = RUN_DIR / 'Experiments.zip'
    with zipfile.ZipFile(archive_path, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(results.rglob('*')):
            if path.is_file() and 'cache' not in path.relative_to(results).parts:
                archive.write(path, Path('Experiments') / path.relative_to(results))
    print('Arquivo de resultados:', archive_path)
    if DOWNLOAD_RESULTS:
        from google.colab import files
        files.download(str(archive_path))
    ''')
    for i, cell in enumerate(cells):
        cell['id'] = f'colab-{i:03d}'
        cell['metadata'] = {}
        if cell.cell_type == 'code':
            cell.outputs = []
            cell.execution_count = None
    notebook = nbformat.v4.new_notebook(cells=cells, metadata={
        'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
        'language_info': {'name': 'python'}, 'accelerator': 'GPU',
        'colab': {'name': 'Trabalho2_Colab.ipynb', 'provenance': []},
    })
    nbformat.validate(notebook)
    return notebook


if __name__ == '__main__':
    import nbformat
    destination = ROOT / 'Trabalho2_Colab.ipynb'
    nbformat.write(build_notebook(), destination)
    print(destination)
