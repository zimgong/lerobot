# Distributed utils borrowed from GO-1. 
import os
from datetime import timedelta

import deepspeed
import torch
import torch.multiprocessing as mp
from torch import distributed as dist

timeout = timedelta(minutes=60)


def init_dist(launcher, backend="nccl", **kwargs):
    if mp.get_start_method(allow_none=True) is None:
        mp.set_start_method("fork")
    if launcher == "pytorch":
        _init_dist_pytorch(backend, **kwargs)
    else:
        raise ValueError(f"Invalid launcher type: {launcher}")


def _init_dist_pytorch(backend, **kwargs):
    rank = int(os.environ["RANK"])
    num_gpus = torch.cuda.device_count()
    torch.cuda.set_device(rank % num_gpus)
    deepspeed.init_distributed(dist_backend=backend)
