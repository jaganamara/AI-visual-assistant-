"""
Device and Memory Management Module.
Provides hardware detection, VRAM diagnostics, automatic FP16 precision selection,
and CUDA Out-Of-Memory (OOM) recovery for constrained GPUs (e.g. RTX 2050 4GB).
"""

import os
import gc
import logging
from typing import Tuple, Dict, Any, Optional
import torch

logger = logging.getLogger(__name__)

# Designated local cache directory for models to keep storage organized
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODELS_CACHE_DIR = os.path.join(PROJECT_ROOT, "models_cache")
os.makedirs(MODELS_CACHE_DIR, exist_ok=True)


class DeviceManager:
    """Manages execution devices, VRAM usage, and safe OOM recovery."""

    @staticmethod
    def get_preferred_device() -> torch.device:
        """
        Determines the optimal execution device.
        Prefers CUDA if available and healthy, otherwise falls back to CPU.
        """
        if torch.cuda.is_available():
            try:
                # Test basic allocation to ensure CUDA runtime is healthy
                _ = torch.zeros(1, device="cuda")
                return torch.device("cuda")
            except Exception as e:
                logger.warning(f"CUDA is available but initialization failed ({e}). Falling back to CPU.")
                return torch.device("cpu")
        return torch.device("cpu")

    @staticmethod
    def get_vram_info() -> Dict[str, Any]:
        """
        Returns diagnostic information about current GPU VRAM utilization.
        """
        if not torch.cuda.is_available():
            return {
                "cuda_available": False,
                "device_name": "CPU",
                "allocated_mb": 0.0,
                "reserved_mb": 0.0,
                "total_vram_mb": 0.0,
                "free_vram_mb": 0.0,
            }

        device_idx = torch.cuda.current_device()
        device_name = torch.cuda.get_device_name(device_idx)
        total_vram = torch.cuda.get_device_properties(device_idx).total_memory / (1024 ** 2)
        allocated = torch.cuda.memory_allocated(device_idx) / (1024 ** 2)
        reserved = torch.cuda.memory_reserved(device_idx) / (1024 ** 2)
        free_approx = total_vram - reserved

        return {
            "cuda_available": True,
            "device_name": device_name,
            "allocated_mb": round(allocated, 2),
            "reserved_mb": round(reserved, 2),
            "total_vram_mb": round(total_vram, 2),
            "free_vram_mb": round(free_approx, 2),
        }

    @staticmethod
    def clear_vram_cache() -> None:
        """
        Evicts unused cached allocations from GPU VRAM and runs Python garbage collection.
        Crucial for 4GB VRAM environments between model transitions.
        """
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            logger.info("VRAM cache cleared successfully.")

    @staticmethod
    def get_optimal_dtype(device: torch.device) -> torch.dtype:
        """
        Returns float16 for CUDA to reduce VRAM pressure by 50%, or float32 for CPU.
        """
        if device.type == "cuda":
            return torch.float16
        return torch.float32

    @staticmethod
    def safe_execute(inference_fn, fallback_fn=None, *args, **kwargs):
        """
        Executes an inference function with automated CUDA OOM error handling and CPU fallback.
        """
        try:
            return inference_fn(*args, **kwargs)
        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            if "out of memory" in str(e).lower() or isinstance(e, torch.cuda.OutOfMemoryError):
                logger.error("CUDA Out-Of-Memory detected! Clearing VRAM and initiating CPU recovery...")
                DeviceManager.clear_vram_cache()
                if fallback_fn is not None:
                    return fallback_fn(*args, **kwargs)
                raise RuntimeError("CUDA OOM occurred and no CPU fallback was provided.") from e
            raise e


def get_cache_dir(subfolder: Optional[str] = None) -> str:
    """Returns absolute path to documented model cache directory."""
    if subfolder:
        target = os.path.join(MODELS_CACHE_DIR, subfolder)
        os.makedirs(target, exist_ok=True)
        return target
    return MODELS_CACHE_DIR
