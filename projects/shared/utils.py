import torch
from typing import Optional
import nvidia_smi

def get_gpu_with_most_memory(
    gpus_to_limit_to: Optional[list[int]] = None,
) -> torch.device:
    if not torch.cuda.is_available():
        if torch.backends.mps.is_available():  # mac native gpu
            return torch.device("mps")
        return torch.device("cpu")
    nvidia_smi.nvmlInit()
    device_count = nvidia_smi.nvmlDeviceGetCount()
    max_free_memory = 0
    chosen_device = 0

    gpu_ids = (
        range(device_count)
        if gpus_to_limit_to is None
        else [id for id in gpus_to_limit_to if id < device_count]
    )
    for i in gpu_ids:
        handle = nvidia_smi.nvmlDeviceGetHandleByIndex(i)
        info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
        if info.free > max_free_memory:
            max_free_memory = info.free
            chosen_device = i

    nvidia_smi.nvmlShutdown()
    return torch.device(f"cuda:{chosen_device}")

def sanezip(x, y):
    """
    Zip, but it errors if the iterators have different lengths.
    """
    iter1 = iter(x)
    iter2 = iter(y)
    while True:
        past_iter_1 = False
        try:
            iter1_next = next(iter1)
            past_iter_1 = True
            iter2_next = next(iter2)

            yield iter1_next, iter2_next
        except StopIteration:
            if past_iter_1:
                # stopped on 2, but not on 1
                raise ValueError("Iterables have different lengths")
            else:
                try:
                    # stopped on 1, but not on 2
                    next(iter2)
                    raise ValueError("Iterables have different lengths")
                except StopIteration:
                    return