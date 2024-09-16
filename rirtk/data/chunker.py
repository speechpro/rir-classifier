import random
import numpy as np
from rirtk.data.classes import Classes


class Chunker:
    def __init__(
            self,
            classes: Classes,
            chunk_size: int,
            chunk_shift: int,
            rand_start: bool,
            num_per_utt: int,
    ):
        self.classes = classes
        self.silence_id = classes.silence_id
        self.chunk_size = chunk_size
        self.chunk_shift = chunk_shift
        self.rand_start = rand_start
        self.num_per_utt = num_per_utt

    def num_chunks(self, length):
        if length < self.chunk_size:
            return 0
        else:
            return (length - self.chunk_size) // self.chunk_shift + 1

    def chunk(self, shift, feats, align=None):
        chunk_feats = feats[shift: shift + self.chunk_size]
        chunk_align = None if align is None else align[shift: shift + self.chunk_size]
        return chunk_feats, chunk_align

    def __call__(self, utid, feats, align):
        if self.rand_start and (len(feats) > self.chunk_shift):
            shift = random.randrange(self.chunk_shift)
            feats = feats[shift:]
            align = align[shift:]
        num_total = self.num_chunks(len(feats))
        chunks = list()
        if num_total > 0:
            num_make = min(num_total, self.num_per_utt)
            shifts = random.sample(range(num_total), num_make)
            class_id = self.classes.id_by_utid(utid)
            half_size = self.chunk_size // 2
            for shift in shifts:
                chunk_feats, chunk_align = self.chunk(
                    shift=shift * self.chunk_shift,
                    feats=feats,
                    align=align
                )
                if np.count_nonzero(chunk_align < 0) > 0:
                    continue
                label = self.silence_id if np.sum(chunk_align) < half_size else class_id
                chunks.append((chunk_feats, label, shift))
        return chunks
