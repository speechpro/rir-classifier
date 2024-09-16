import re
import torch
import random
import logging
import numpy as np
from tqdm import tqdm
from torchaudio.functional import convolve
from typing import Dict, List, Tuple, Union
from rirtk.utils.input_output import int16_to_float32, MemDataSet, SeqDataSet


class Segment:
    def __init__(self, speech: bool, begin: int, end: int):
        self.speech = speech
        self.begin = begin
        self.end = end

    @property
    def length(self) -> int:
        return self.end - self.begin


class AlignSet:
    def __init__(
            self,
            align_set: Dict[str, np.ndarray[Union[int, float]]],
            frame_len: int,
            frame_step: int,
    ):
        logging.info('Converting alignments')
        self.data = dict()
        for utid, align in tqdm(align_set.items(), mininterval=3, desc='Converting alignments'):
            segms = list()
            for i, label in enumerate(align):
                speech = (label > 0.5)
                if (len(segms) == 0) or (speech != segms[-1].speech):
                    segms.append(Segment(speech=speech, begin=i, end=i + 1))
                else:
                    segms[-1].end += 1
            # num = (wave - len) // step + 1
            # wave = (num - 1) * step + len
            for segm in segms:
                length = (segm.length - 1) * frame_step + frame_len
                segm.begin *= frame_step
                segm.end = segm.begin + length
            self.data[utid] = segms
        logging.info(f'  converted {len(self.data)} alignments')

    def __contains__(self, utid: str) -> bool:
        return utid in self.data

    def __getitem__(self, utid: str) -> List[Segment]:
        return self.data[utid]

    def __len__(self) -> int:
        return len(self.data)


class NoiseSet:
    def __init__(self, frequency: int, noise_set: MemDataSet):
        self.data = list()
        for nsid, (freq, samps) in noise_set:
            assert freq == frequency, f'Wrong frequency {freq} (must be {frequency}) for noise {nsid}'
            samps = int16_to_float32(samps)
            self.data.append((nsid, samps))

    def items(self):
        while True:
            random.shuffle(self.data)
            for nsid, samps in self.data:
                yield nsid, samps


class RirAndWave:
    def __init__(
            self,
            utt_id: str,
            rir_id: str,
            speech: bool,
            rir_samps: np.ndarray,
            wav_samps: np.ndarray,
            vad_align: List[Segment],
    ):
        self.utt_id = utt_id
        self.rir_id = rir_id
        self.speech = speech
        self.rir_samps = rir_samps
        self.rir_length = len(rir_samps)
        self.wav_samps = wav_samps
        self.wav_length = len(wav_samps)
        self.vad_align = vad_align

    def set_wave(self, wav_samps):
        self.wav_samps = wav_samps
        self.wav_length = len(wav_samps)

    def power(self):
        length = self.wav_length
        if self.vad_align is None:
            return np.mean(np.square(self.wav_samps[0: length]))
        else:
            power_sum = 0.0
            power_len = 0
            for segm in self.vad_align:
                if segm.speech:
                    if segm.begin > length:
                        break
                    end = min(segm.end, length)
                    power_sum += np.sum(np.square(self.wav_samps[segm.begin: end]))
                    power_len += end - segm.begin
            if power_len == 0:
                return np.mean(np.square(self.wav_samps[0: length]))
            else:
                return power_sum / power_len


def rir_convolve(
        device: torch.device,
        batch: List[RirAndWave]
) -> List[RirAndWave]:
    max_rir_len = 0
    max_wav_len = 0
    for item in batch:
        if max_rir_len < item.rir_length:
            max_rir_len = item.rir_length
        if item.speech and (max_wav_len < item.wav_length):
            max_wav_len = item.wav_length
    batch_size = len(batch)
    rir_batch = np.zeros((batch_size, max_rir_len), dtype=np.float32)
    for i, item in enumerate(batch):
        rir_batch[i, 0: item.rir_length] = item.rir_samps
    rir_batch /= np.max(np.abs(rir_batch), axis=1, keepdims=True)
    rir_batch = torch.tensor(rir_batch, device=device)
    wav_batch = np.zeros((batch_size, max_wav_len), dtype=np.float32)
    for i, item in enumerate(batch):
        wav_batch[i, 0: item.wav_length] = item.wav_samps
    wav_batch /= np.max(np.abs(wav_batch), axis=1, keepdims=True)
    wav_batch = torch.tensor(wav_batch, device=device)
    convolved = convolve(wav_batch, rir_batch, mode='full').cpu().numpy()
    for i, item in enumerate(batch):
        shift = np.argmax(item.rir_samps)
        item.set_wave(convolved[i, shift: shift + item.wav_length])
    return batch


def make_mixtures(
        snr_range: List[float],
        batch: List[RirAndWave]
) -> List[Tuple[str, np.ndarray]]:
    by_utts = dict()
    for item in batch:
        if item.utt_id in by_utts:
            by_utts[item.utt_id][item.speech] = item
        else:
            by_utts[item.utt_id] = {item.speech: item}
    mixed_waves = list()
    for utt_id, utt_set in by_utts.items():
        assert True in utt_set, f'Failed to find speech for utterance {utt_id}'
        spc_item = utt_set[True]
        assert False in utt_set, f'Failed to find noise for utterance {utt_id}'
        nse_item = utt_set[False]
        # ============== DEBUG ==============
        assert spc_item.wav_length == len(spc_item.wav_samps), f'Wrong wave length stored {spc_item.wav_length} (actual value is {len(spc_item.wav_samps)}) for utterance {utt_id}'
        assert nse_item.wav_length == len(nse_item.wav_samps), f'Wrong wave length stored {nse_item.wav_length} (actual value is {len(nse_item.wav_samps)}) for utterance {utt_id}'
        assert nse_item.wav_length == spc_item.wav_length, f'Wrong noise length {nse_item.wav_length} (must be {spc_item.wav_length}) for utterance {utt_id}'
        # ===================================
        spc_power = spc_item.power()
        nse_power = nse_item.power()
        # snr = 10 * np.log10(ps / pn)
        snr_value = random.uniform(*snr_range)
        snr_scale = np.sqrt(spc_power / (np.power(10, snr_value / 10) * nse_power))
        nse_item.wav_samps *= snr_scale
        # ============== DEBUG ==============
        nse_power = nse_item.power()
        snr = 10 * np.log10(spc_power / nse_power)
        diff = abs((snr - snr_value) / snr_value)
        assert diff < 1e-5
        # ===================================
        mix_samps = spc_item.wav_samps + nse_item.wav_samps
        nrm_scale = max(abs(mix_samps.min()), abs(mix_samps.max()))
        mix_samps /= nrm_scale
        mix_id = f'{utt_id}#{spc_item.rir_id}%{snr_value:.2f}'
        mixed_waves.append((mix_id, mix_samps))
    return mixed_waves


def reverberate(
        device: torch.device,
        rir_set: MemDataSet,
        noise_set: MemDataSet,
        # align_set: Dict[str, np.ndarray[Union[int, float]]],
        align_set: Dict[str, Segment],
        wave_set: SeqDataSet,
        frequency: int,
        # frame_len: float,
        # frame_step: float,
        tolerance: float,
        snr_range: List[float],
        num_repeats: int,
        batch_size: int,
        align_from_utid: str,
        room_from_ririd: str,
):
    # frame_len = round(frequency * frame_len / 1000)
    # frame_step = round(frequency * frame_step / 1000)
    tolerance = round(frequency * tolerance / 1000)

    align_from_utid = re.compile(align_from_utid)
    room_from_ririd = re.compile(room_from_ririd)

    rirs_unodered = list()
    rirs_by_room = dict()
    for rir_id, (freq, samps) in rir_set:
        assert freq == frequency, f'Wrong frequency {freq} (must be {frequency}) for RIR {rir_id}'
        samps = int16_to_float32(samps)
        pair = (rir_id, samps)
        rirs_unodered.append(pair)
        match = room_from_ririd.match(rir_id)
        assert match is not None, f'Failed to get room ID from RIR ID {rir_id} with regular expression {room_from_ririd}'
        room = match.group(1)
        if room in rirs_by_room:
            rirs_by_room[room].append(pair)
        else:
            rirs_by_room[room] = [pair]

    # logging.info('Converting alignments')
    # align_set = AlignSet(align_set, frame_len, frame_step)
    # logging.info(f'  converted {len(align_set)} alignments')

    noise_set = NoiseSet(frequency, noise_set)
    noise_set = iter(noise_set.items())

    wave_set = iter(wave_set)

    num_done = 0
    waves: List = next(wave_set)
    for _ in range(num_repeats):
        random.shuffle(rirs_unodered)
        batch = list()
        for rir_id, rir_speech in rirs_unodered:
            match = room_from_ririd.match(rir_id)
            assert match is not None, f'Failed to get room ID from RIR ID {rir_id} with regular expression {room_from_ririd}'
            room_id = match.group(1)
            while True:
                if len(waves) == 0:
                    waves = next(wave_set)
                utt_id, (freq, wav_speech) = waves.pop(0)
                assert freq == frequency, f'Wrong frequency {freq} (must be {frequency}) for recording {utt_id}'
                match = align_from_utid.match(utt_id)
                assert match is not None, f'Failed to get VAD align ID from utterance ID {utt_id} with regular expression {align_from_utid}'
                vad_id = match.group(1)
                if vad_id in align_set:
                    break
            wav_speech = int16_to_float32(wav_speech)
            vad_align = align_set[vad_id]
            wav_len = len(wav_speech)
            ali_len = vad_align[-1].end
            diff = abs(ali_len - wav_len)
            assert diff <= tolerance, f'Wrong alignment length {ali_len} (must be {wav_len}) for utterance {utt_id}'
            batch_item = RirAndWave(
                utt_id=utt_id,
                rir_id=rir_id,
                speech=True,
                rir_samps=rir_speech,
                wav_samps=wav_speech,
                vad_align=vad_align,
            )
            batch.append(batch_item)
            rir_noise_id, rir_noise = random.choice(rirs_by_room[room_id])
            noise_id, wav_noise = next(noise_set)
            noise_len = len(wav_noise)
            count = (wav_len - 1) // noise_len + 1
            if count > 1:
                wav_noise = [wav_noise] * count
                wav_noise = np.hstack(wav_noise)
                noise_len = len(wav_noise)
            shift = noise_len - wav_len
            if shift > 0:
                shift = random.randrange(shift)
            wav_noise = wav_noise[shift: shift + wav_len]
            noise_len = len(wav_noise)
            assert noise_len == wav_len, f'Wrong noise length {noise_len} (must be {wav_len}) for utterance {utt_id}'
            batch_item = RirAndWave(
                utt_id=utt_id,
                rir_id=rir_id,
                speech=False,
                rir_samps=rir_noise,
                wav_samps=wav_noise,
                vad_align=vad_align,
            )
            batch.append(batch_item)
            if (len(batch) // 2) < batch_size:
                continue
            try:
                batch = rir_convolve(device, batch)
            except BaseException as exception:
                logging.error(f'Failed to convolve {len(batch)} waves in batch:\n{exception}')
                continue
            mixed = make_mixtures(snr_range, batch)
            for mix_id, mix_samps in mixed:
                yield mix_id, (frequency, mix_samps)
            batch = list()
            num_done += len(mixed)
    logging.info(f'Successfully created {num_done} mixtures')
