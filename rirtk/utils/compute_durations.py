import logging
import torchaudio
from tqdm import tqdm
from pathlib import Path


def compute_durations(path_scp: str, path_dur: str) -> None:
    logging.info(f'Computing wav durations for:\n{path_scp}')
    path_scp = Path(path_scp)
    assert path_scp.is_file(), f'File {path_scp} does not exist'
    lines = path_scp.read_text(encoding='utf-8').strip().split('\n')
    lines = [line.split() for line in lines]
    utt2dur = list()
    for utid, path in tqdm(lines, mininterval=3, desc='Computing durations'):
        path = Path(path)
        assert path.is_file(), f'File {path} does not exist'
        meta = torchaudio.info(path)
        udur = meta.num_frames / meta.sample_rate
        utt2dur.append(f'{utid} {udur:.2f}')
    path_dur = Path(path_dur).absolute()
    if not path_dur.parent.exists():
        logging.info(f'Creating directory {path_dur.parent}')
        path_dur.parent.mkdir(parents=True, exist_ok=True)
    logging.info(f'Writing file {path_dur}')
    path_dur.write_text('\n'.join(utt2dur) + '\n', encoding='utf-8')
