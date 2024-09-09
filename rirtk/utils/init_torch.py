import torch
import random
import numpy as np
from datetime import datetime


def seed_from_time():
    seed = datetime.now().timestamp()
    return int(1e9 * np.modf(seed)[0])


def set_random_seed(seed):
    if seed < 0:
        seed = seed_from_time()
    random.seed(seed)
    np.random.seed(seed)
    torch.random.manual_seed(seed)


def init_cudnn(cudnn_enabled, cudnn_benchmark, cudnn_deterministic):
    torch.backends.cudnn.enabled = cudnn_enabled
    torch.backends.cudnn.benchmark = cudnn_benchmark
    torch.backends.cudnn.deterministic = cudnn_deterministic


class WorkerInitFn:
    def __init__(self, seed, epoch=0):
        self.seed = seed if seed < 0 else seed + epoch

    def __call__(self, worker_id):
        seed = seed_from_time() + worker_id if self.seed < 0 else self.seed + worker_id
        set_random_seed(seed)
