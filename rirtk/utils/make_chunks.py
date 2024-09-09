import re
import logging
import numpy as np
from typing import Dict
from rirtk.data.chunker import Chunker


def make_chunks(
        chunker: Chunker,
        align_set: Dict[str, np.ndarray],
        feats_set: Dict[str, np.ndarray],
        align_from_utid: str = None,
        tolerance: int = 0,
):
    if align_from_utid is not None:
        align_from_utid = re.compile(align_from_utid)
    num_done = 0
    num_skip = 0
    for feats_utid, feats in feats_set:
        if align_from_utid is None:
            align_utid = feats_utid
        else:
            match = align_from_utid.match(feats_utid)
            assert match is not None, f'Failed to get VAD align ID from utterance ID {feats_utid} with regular expression {align_from_utid}'
            align_utid = match.group(1)
        if align_utid not in align_set:
            num_skip += 1
            continue
        align = align_set[align_utid]
        feats_len = len(feats)
        align_len = len(align)
        diff = abs(align_len - feats_len)
        assert diff <= tolerance, f'Wrong alignment length {align_len} (must be {feats_len}) for utterance {align_utid}'
        if diff > 0:
            length = min(feats_len, align_len)
            feats = feats[0: length]
            align = align[0: length]
        inputs, labels = chunker(feats_utid, feats, align)
        for chunk, label in zip(inputs, labels):
            yield f'{feats_utid}^{label}', chunk
        num_done += len(inputs)
    logging.info(f'Generated {num_done} chunks')
