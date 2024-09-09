import numpy as np


def set_max_level(wave_set, max_level):
    for utid, (freq, samps) in wave_set:
        if samps.dtype != np.float32:
            samps = samps.astype(np.float32)
        samps -= np.mean(samps)
        level = max(abs(samps.min()), abs(samps.max()))
        samps *= max_level / (level * abs(np.iinfo(np.int16).min))
        # scale = max_level / level
        # samps = (scale * samps).round().astype(np.int16)
        yield utid, (freq, samps)
