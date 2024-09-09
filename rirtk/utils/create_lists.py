import logging
from pathlib import Path
from typing import Union, List
from rirtk.utils.input_output import list_files
from inex.helpers import check_md5_hash, check_existence
from rirtk.data.classes import Classes


def create_scp(
        pathnames: Union[str, List[str]],
        path_scp: str,
        recursive=True,
) -> None:
    logging.info(f'Creating file list from:\n{pathnames}')
    paths = list_files(pathnames, recursive=recursive)
    paths = list(sorted(paths))
    logging.info(f'Created list of {len(paths)} paths')
    lines = [f'{Path(path).stem} {path}' for path in paths]
    path_scp = Path(path_scp).absolute()
    if not path_scp.parent.exists():
        logging.info(f'Creating directory {path_scp.parent}')
        path_scp.parent.mkdir(parents=True, exist_ok=True)
    logging.info(f'Writing file {path_scp}')
    path_scp.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def create_rir_scp(
        pathnames: Union[str, List[str]],
        path_scp: str,
        recursive=True,
) -> None:
    logging.info(f'Creating file list from:\n{pathnames}')
    paths = list_files(pathnames, recursive=recursive)
    paths = list(sorted(paths))
    logging.info(f'Created list of {len(paths)} paths')
    rtypes = {'smallroom', 'mediumroom', 'largeroom'}
    lines = list()
    for path in paths:
        path = Path(path).absolute()
        rtype = str(path.parent.parent.name)
        assert rtype in rtypes, f'Wrong RIR room type {rtype} (must be {rtypes}) for path {path}'
        lines.append(f'{rtype[0]}{path.stem} {path}')
    path_scp = Path(path_scp).absolute()
    if not path_scp.parent.exists():
        logging.info(f'Creating directory {path_scp.parent}')
        path_scp.parent.mkdir(parents=True, exist_ok=True)
    logging.info(f'Writing file {path_scp}')
    path_scp.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def create_musan_scp(
        musan_root: str,
        musan_scp: str,
) -> None:
    musan_root = Path(musan_root).absolute()
    wave_root = musan_root / 'noise' / 'free-sound'
    path = wave_root / 'ANNOTATIONS'
    logging.info(f'Creating file list from:\n{path}')
    check_md5_hash({'path': path, 'md5': 'dfe81524a7b0f10491777da923b39bc3'})
    names = path.read_text(encoding='utf-8').strip().split('\n')[1:]
    paths = [str(wave_root / f'{name}.wav') for name in names]
    paths = list(sorted(paths))
    logging.info(f'Created list of {len(paths)} paths')
    check_existence(paths)
    lines = [f'{Path(path).stem} {path}' for path in paths]
    musan_scp = Path(musan_scp).absolute()
    if not musan_scp.parent.exists():
        logging.info(f'Creating directory {musan_scp.parent}')
        musan_scp.parent.mkdir(parents=True, exist_ok=True)
    logging.info(f'Writing file {musan_scp}')
    musan_scp.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def create_classes(
        path_scp: str,
        path_classes: str,
) -> None:
    logging.info(f'Creating classes from:\n{path_scp}')
    path_scp = Path(path_scp)
    assert path_scp.is_file(), f'File {path_scp} does not exist'
    lines = path_scp.read_text(encoding='utf-8').strip().split('\n')
    classes = {line.strip().split()[0] for line in lines}
    classes = list(sorted(classes))
    classes.append(Classes.silence_name)
    logging.info(f'Number of classes is {len(classes)}')
    path_classes = Path(path_classes).absolute()
    if not path_classes.parent.exists():
        logging.info(f'Creating directory {path_classes.parent}')
        path_classes.parent.mkdir(parents=True, exist_ok=True)
    logging.info(f'Writing file {path_classes}')
    path_classes.write_text('\n'.join(classes) + '\n', encoding='utf-8')
