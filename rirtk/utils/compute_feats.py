import torch
import numpy as np
from typing import Dict
from torchaudio.compliance import kaldi


def compute_fbank(wave_set: Dict, **kwargs):
    for utid, (freq, samps) in wave_set:
        assert samps.dtype == np.float32, f'Wrong wave form samples type {samps.dtype} (must be np.float32)'
        if 'sample_frequency' in kwargs:
            frequency = kwargs['sample_frequency']
            assert freq == frequency, f'Wrong sampling frequency {freq} (must be {frequency}) in utterance {utid}'
        else:
            kwargs['sample_frequency'] = freq
        if samps.ndim == 1:
            samps = np.expand_dims(samps, axis=0)
        feats = kaldi.fbank(torch.tensor(samps), **kwargs).numpy()
        yield utid, feats
