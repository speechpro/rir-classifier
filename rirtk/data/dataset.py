import re
import torch
import einops
import logging
import numpy as np
from pathlib import Path
from typing import Dict, Any
from kaldiio import load_mat
from torch.utils.data import Dataset


class BatchSet(Dataset):
    def __init__(
            self,
            targ_from_chid: str,
            data_dir: str,
            batch_size: int,
    ):
        super().__init__()
        self.targ_from_chid = re.compile(targ_from_chid)
        path = Path(data_dir) / 'chunks_shuf.scp'
        if not path.is_file():
            path = Path(data_dir) / 'chunks.scp'
        logging.debug(f'Loading data references from {path}')
        assert path.is_file(), f'File {path} does not exist'
        lines = path.read_text(encoding='utf-8').strip().split('\n')
        lines = [line.strip().split(maxsplit=1) for line in lines]
        self.data = [(utid, path) for utid, path in lines]
        logging.debug(f'Loaded {len(self.data)} data references')
        self.data = np.array(self.data)
        self.num_batches = len(self.data) // batch_size
        self.data = self.data[0: batch_size * self.num_batches]
        self.data = einops.rearrange(self.data, '(n b) c -> n b c', b=batch_size)
        np.random.shuffle(self.data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item: int) -> Dict[str, Any]:
        chids = list()
        labels = list()
        inputs = list()
        fd_dict = dict()
        for chid, path in self.data[item]:
            chids.append(chid)
            match = self.targ_from_chid.match(chid)
            assert match is not None, f'Failed to parse chunk ID {chid} with regular expression {self.targ_from_chid}'
            labels.append(int(match.group(1)))
            inputs.append(load_mat(path, fd_dict=fd_dict))
        for stream in fd_dict.values():
            stream.close()
        labels = torch.tensor(labels, dtype=torch.int64)
        inputs = torch.tensor(inputs, dtype=torch.float32)
        return {
            'chids': chids,
            'inputs': inputs,
            'labels': labels,
        }


def test_loader(loader):
    count = 0
    for batch in loader:
        count += 1
    logging.info(f'Processed {count} batches')
