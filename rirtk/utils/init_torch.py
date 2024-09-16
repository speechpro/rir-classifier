import torch
import random
import numpy as np
from datetime import datetime


def seed_from_time() -> int:
    seed = datetime.now().timestamp()
    return int(1e9 * np.modf(seed)[0])


def set_random_seed(seed: int) -> None:
    if seed < 0:
        seed = seed_from_time()
    random.seed(seed)
    np.random.seed(seed)
    torch.random.manual_seed(seed)


def init_cudnn(
        cudnn_enabled: bool,
        cudnn_benchmark: bool,
        cudnn_deterministic: bool,
) -> None:
    torch.backends.cudnn.enabled = cudnn_enabled
    torch.backends.cudnn.benchmark = cudnn_benchmark
    torch.backends.cudnn.deterministic = cudnn_deterministic


class WorkerInitFn:
    def __init__(self, seed: int, epoch: int = 0):
        self.seed = seed if seed < 0 else seed + epoch

    def __call__(self, worker_id: int) -> None:
        seed = seed_from_time() + worker_id if self.seed < 0 else self.seed + worker_id
        set_random_seed(seed)
