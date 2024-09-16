import re
import logging
import numpy as np
from rirtk.data.chunker import Chunker
from typing import Iterable, Tuple, List, Dict


def make_chunks(
        chunker: Chunker,
        align_set: Dict[str, np.ndarray],
        feats_set: Iterable[Tuple[str, np.ndarray]],
        align_from_utid: str = None,
        num_chunks: int = None,
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
        chunks = chunker(feats_utid, feats, align)
        for chunk, label, shift in chunks:
            yield f'{feats_utid}^{shift}@{label}', chunk
            num_done += 1
            if (num_chunks is not None) and (num_done == num_chunks):
                break
        if (num_chunks is not None) and (num_done == num_chunks):
            break
    logging.info(f'Generated {num_done} chunks')


def unfold_batches(
        data_set: Iterable[List[Tuple[str, np.ndarray[np.float32]]]],
) -> Iterable[Tuple[str, np.ndarray[np.float32]]]:
    for batch in data_set:
        for utid, chunk in batch:
            yield utid, chunk
