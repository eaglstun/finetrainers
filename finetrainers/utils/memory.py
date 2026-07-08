import gc
from typing import Any, Dict, Union

import torch

from finetrainers.logging import get_logger


logger = get_logger()


def get_memory_statistics(precision: int = 3) -> Dict[str, Any]:
    memory_allocated = None
    memory_reserved = None
    max_memory_allocated = None
    max_memory_reserved = None

    if torch.cuda.is_available():
        device = torch.cuda.current_device()
        memory_allocated = torch.cuda.memory_allocated(device)
        memory_reserved = torch.cuda.memory_reserved(device)
        max_memory_allocated = torch.cuda.max_memory_allocated(device)
        max_memory_reserved = torch.cuda.max_memory_reserved(device)

    elif torch.backends.mps.is_available():
        memory_allocated = torch.mps.current_allocated_memory()
        memory_reserved = torch.mps.driver_allocated_memory()

    else:
        logger.warning("No CUDA, MPS, or ROCm device found. Memory statistics are not available.")

    def _to_gb(x):
        return round(bytes_to_gigabytes(x), ndigits=precision) if x is not None else None

    return {
        "memory_allocated": _to_gb(memory_allocated),
        "memory_reserved": _to_gb(memory_reserved),
        "max_memory_allocated": _to_gb(max_memory_allocated),
        "max_memory_reserved": _to_gb(max_memory_reserved),
    }


def bytes_to_gigabytes(x: int) -> float:
    if x is not None:
        return x / 1024**3


def free_memory() -> None:
    if torch.cuda.is_available():
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    elif torch.backends.mps.is_available():
        gc.collect()
        torch.mps.empty_cache()

    # TODO(aryan): handle non-cuda devices


def reset_peak_memory_stats(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    # torch.mps has no reset_peak_memory_stats; peak tracking is CUDA-only for now


def make_contiguous(x: Union[torch.Tensor, Dict[str, torch.Tensor]]) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
    if isinstance(x, torch.Tensor):
        return x.contiguous()
    elif isinstance(x, dict):
        return {k: make_contiguous(v) for k, v in x.items()}
    else:
        return x
