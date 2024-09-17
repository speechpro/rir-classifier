import os
import gzip
import glob
import logging
import torchaudio
import numpy as np
from tqdm import tqdm
from pathlib import Path
from inex.helpers import OptionalFile
from kaldiio import ReadHelper, WriteHelper, load_mat
from torch.utils.data import Dataset, IterableDataset
from kaldiio.compression_header import kSpeechFeature
from typing import Optional, Union, List, Dict, Iterable, Tuple


def int16_to_float32(samples: np.ndarray[np.int16]) -> np.ndarray[np.float32]:
    if samples.dtype == np.int16:
        samples = samples.astype(np.float32)
        samples /= np.abs(np.iinfo(np.int16).min)
    return samples


def float32_to_int16(samples: np.ndarray[np.float32]) -> np.ndarray[np.int16]:
    if samples.dtype == np.float32:
        max_value = max(abs(samples.min()), abs(samples.max()))
        assert max_value < 1.00001, f'Wring maximum absolute sample value {max_value} (must be less or equal 1)'
        samples = (samples * np.abs(np.iinfo(np.int16).min)).round().astype(np.int16)
    return samples


def list_files(pathnames: Union[str, List[str]], recursive=True) -> List[str]:
    if isinstance(pathnames, str):
        pathnames = [pathnames]
    paths = set()
    for pathname in pathnames:
        for path in glob.iglob(pathname, recursive=recursive):
            path = Path(path).absolute()
            paths.add(str(path))
    return list(paths)


def read_vectors_from_text_ark(
        pathnames: Union[str, List[str]],
        recursive: bool = True,
        dtype=np.int32,
) -> Dict[str, np.ndarray]:
    if isinstance(pathnames, str):
        pathnames = [pathnames]
    logging.debug(f'Loading vectors from\n{pathnames}')
    count = 0
    data = dict()
    for pathname in pathnames:
        for path in glob.iglob(pathname, recursive=recursive):
            with OptionalFile(path) as stream:
                lines = stream.readlines()
            for line in lines:
                line = line.strip()
                if len(line) == 0:
                    continue
                index = line.find(' ')
                assert index > 0, f'Failed to find utterance ID in line "{line}" in file {path}'
                utid = line[0: index]
                index1 = line[index: index + 10].find('[')
                if index1 < 0:
                    align = line[index + 1:]
                    align = np.fromstring(align, dtype=dtype, sep=' ')
                else:
                    index1 += index
                    index2 = line.rfind(']')
                    assert index2 > 0, f'Failed to find data end symbol in line "{line}" in file {path}'
                    align = line[index1 + 1: index2]
                    align = np.fromstring(align, dtype=dtype, sep=' ')
                data[utid] = align
            count += 1
    assert count > 0, f'File(s) {pathnames} does not exist'
    logging.debug(f'  loaded {len(data)} vectors from {count} files')
    return data


class AudioSet(IterableDataset):
    def __init__(
            self,
            wav_scp: str,
            utt2dur: Optional[str] = None,
            reverse: bool = True,
            num_loops: Union[bool, int] = None,
    ):
        super().__init__()
        logging.debug(f'Loading file list from {wav_scp}')
        wav_scp = Path(wav_scp)
        assert wav_scp.is_file(), f'File {wav_scp} does not exist'
        self.paths = wav_scp.read_text(encoding='utf-8').strip().split('\n')
        self.paths = [line.split(maxsplit=1) for line in self.paths]
        logging.debug(f'Loaded {len(self.paths)} paths')
        if utt2dur is not None:
            logging.debug(f'Loading durations from {utt2dur}')
            utt2dur = Path(utt2dur)
            assert utt2dur.is_file(), f'File {utt2dur} does not exist'
            utt2dur = utt2dur.read_text(encoding='utf-8').strip().split('\n')
            utt2dur = [line.split() for line in utt2dur]
            utt2dur = {utid: duration for utid, duration in utt2dur}
            logging.debug(f'Loaded {len(utt2dur)} durations')
            self.paths = [(utt2dur[utid], utid, path) for utid, path in self.paths]
            self.paths = list(sorted(self.paths, reverse=reverse))
            self.paths = [(utid, path) for _, utid, path in self.paths]
        self.num_loops = num_loops

    def items(self) -> Iterable[Tuple[str, Tuple[int, np.ndarray]]]:
        num_done = 0
        num_loops = 0
        while True:
            for utid, path in self.paths:
                path = Path(path).absolute()
                assert path.is_file(), f'File {path} does not exist'
                data, freq = torchaudio.load(path)
                data = data.numpy()
                if (data.ndim == 2) and (len(data) == 1):
                    data = data[0]
                yield utid, (freq, data)
                num_done += 1
            if self.num_loops is None:
                break
            num_loops += 1
            if isinstance(self.num_loops, bool):
                if not self.num_loops:
                    break
            elif num_loops == self.num_loops:
                break
        logging.debug(f'Processed {num_done} utterances')

    def __iter__(self):
        return iter(self.items())

    def __call__(self, batch):
        return batch


class LazyFeatsSet(Dataset):
    def __init__(self, feats_scp: str):
        super().__init__()
        logging.debug(f'Loading features references from {feats_scp}')
        feats_scp = Path(feats_scp)
        assert feats_scp.is_file(), f'File {feats_scp} does not exist'
        lines = feats_scp.read_text(encoding='utf-8').strip().split('\n')
        lines = [line.strip().split(maxsplit=1) for line in lines]
        self.data = [(utid, path) for utid, path in lines]
        logging.debug(f'Loaded {len(self.data)} features references')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item: int) -> Tuple[str, np.ndarray[np.float32]]:
        utid, path = self.data[item]
        feats = load_mat(path)
        return utid, feats

    def __call__(
            self,
            batch: List[Tuple[str, np.ndarray[np.float32]]],
    ) -> List[Tuple[str, np.ndarray[np.float32]]]:
        return batch


class SeqDataSet:
    def __init__(
            self,
            pathname: Union[str, List[str]],
            recursive: bool = False,
            num_loops: bool = None,
    ):
        if isinstance(pathname, str):
            pathname = [pathname]
        self.paths = list()
        for path_mask in pathname:
            count = 0
            if path_mask.startswith('ark:') or path_mask.startswith('scp:'):
                logging.debug(f'Data source is {path_mask}')
                self.paths.append(path_mask)
                count += 1
            else:
                logging.debug(f'Loading data paths by mask {path_mask}')
                for path in glob.iglob(path_mask, recursive=recursive):
                    self.paths.append(path)
                    count += 1
                assert count > 0, f'Failed to find any data files by mask {path_mask}'
        logging.debug(f'  found {len(self.paths)} data files')
        self.num_loops = num_loops

    def items(self):
        num_done = 0
        num_loops = 0
        while True:
            for path in self.paths:
                if path.startswith('ark:') or path.startswith('scp:'):
                    rspec = path
                else:
                    assert os.path.isfile(path), f'File {path} does not exist'
                    rspec = f'scp:{path}' if path.endswith('.scp') else f'ark:{path}'
                logging.debug(f'Reading data from {rspec}')
                with ReadHelper(rspec) as reader:
                    for utid, data in reader:
                        yield utid, data
                        num_done += 1
            if self.num_loops is None:
                break
            num_loops += 1
            if isinstance(self.num_loops, bool):
                if not self.num_loops:
                    break
            elif num_loops == self.num_loops:
                break
        logging.debug(f'Processed {num_done} utterances')

    def __iter__(self):
        return iter(self.items())


class MemDataSet:
    def __init__(
            self,
            pathname: Union[str, List[str]],
            compress: bool = False,
            recursive: bool = False,
    ):
        if isinstance(pathname, str):
            pathname = [pathname]
        paths = list()
        for path_mask in pathname:
            count = 0
            if path_mask.startswith('ark:') or path_mask.startswith('scp:'):
                logging.debug(f'Data source is {path_mask}')
                paths.append(path_mask)
                count += 1
            else:
                logging.debug(f'Loading data paths by mask {path_mask}')
                for path in glob.iglob(path_mask, recursive=recursive):
                    paths.append(path)
                    count += 1
                assert count > 0, f'Failed to find any data files by mask {path_mask}'
        logging.debug(f'  found {len(paths)} data files')
        self.data = dict()
        num_done = 0
        for path in paths:
            if path.startswith('ark:') or path.startswith('scp:'):
                rspec = path
            else:
                assert os.path.isfile(path), f'File {path} does not exist'
                rspec = f'scp:{path}' if path.endswith('.scp') else f'ark:{path}'
            logging.debug(f'Reading data from {rspec}')
            with ReadHelper(rspec) as reader:
                for utid, data in reader:
                    if compress:
                        data = (gzip.compress(data.tobytes(), compresslevel=9), data.shape, data.dtype)
                    self.data[utid] = data
                    num_done += 1
        self.compress = compress
        logging.debug(f'Loaded data for {num_done} utterances')

    def __contains__(self, utid) -> bool:
        return utid in self.data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, utid) -> Optional[np.ndarray]:
        if utid in self.data:
            data = self.data[utid]
            if self.compress:
                data, shape, dtype = data
                data = np.frombuffer(gzip.decompress(data), dtype=dtype)
                data.resize(shape)
            return data
        else:
            return None

    def keys(self):
        return self.data.keys()

    def items(self):
        for utid, data in self.data.items():
            if self.compress:
                data, shape, dtype = data
                data = np.frombuffer(gzip.decompress(data), dtype=dtype)
                data.resize(shape)
            yield utid, data

    def __iter__(self):
        return iter(self.items())


def write_data(
        data_set,
        directory=None,
        write_spec=None,
        file_stem='feats',
        archive_index=None,
        compress=False,
        write_function=None,
        disable_tqdm=False,
        tqdm_desc='Writing data',
):
    if directory is not None:
        directory = Path(directory).absolute()
        if not directory.exists():
            logging.info(f'Creating directory {directory}')
            directory.mkdir(parents=True, exist_ok=True)
        if write_spec is None:
            if archive_index is None:
                write_spec = f'{directory}/{file_stem}'
            else:
                write_spec = f'{directory}/{file_stem}.{archive_index}'
            write_spec = f'ark,scp:{write_spec}.ark,{write_spec}.scp'
    if write_spec is None:
        write_spec = 'ark:/dev/null'
    comp_method = kSpeechFeature if compress else None
    logging.info(f'Writing data to {write_spec}')
    num_done = 0
    with WriteHelper(write_spec, compression_method=comp_method, write_function=write_function) as writer:
        for utid, data in tqdm(data_set, mininterval=3, disable=disable_tqdm, desc=tqdm_desc):
            writer(utid, data)
            num_done += 1
    logging.info(f'  wrote {num_done} data items')
