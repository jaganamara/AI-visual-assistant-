"""
Environment and Hardware Verification Script.
Checks Python version, PyTorch installation, CUDA acceleration on RTX 2050,
VRAM limits, and reports health.
"""

import sys
import os

# Add project root to sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def run_diagnostics():
    print("=" * 60)
    print("AI-POWERED VISUAL ASSISTANT: ENVIRONMENT DIAGNOSTICS")
    print("=" * 60)
    print(f"Python Executable: {sys.executable}")
    print(f"Python Version   : {sys.version.split()[0]}")

    # Check whether running inside isolated .venv
    in_venv = sys.prefix != sys.base_prefix
    print(f"Isolated .venv   : {'YES (Isolated)' if in_venv else 'NO (Warning: Global Python)'}")

    # Check PyTorch
    try:
        import torch
        print(f"PyTorch Version  : {torch.__version__}")
        cuda_avail = torch.cuda.is_available()
        print(f"CUDA Available   : {cuda_avail}")

        if cuda_avail:
            gpu_name = torch.cuda.get_device_name(0)
            vram_mb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 2)
            print(f"Detected GPU     : {gpu_name}")
            print(f"Dedicated VRAM   : {vram_mb:.0f} MB")
            print(f"CUDA Capability  : {torch.cuda.get_device_capability(0)}")

            # Test tensor allocation
            x = torch.ones((100, 100), device="cuda")
            y = x @ x
            print("CUDA Tensor Test : PASS (GPU compute verified)")
        else:
            print("Note: CUDA not detected, CPU fallback will be active.")

    except ImportError:
        print("PyTorch Version  : NOT INSTALLED")
        return False

    # Check DeviceManager
    try:
        from src.utils.device_manager import DeviceManager
        vram_info = DeviceManager.get_vram_info()
        print(f"DeviceManager    : READY (Device: {DeviceManager.get_preferred_device()})")
        print(f"VRAM Status      : {vram_info}")
    except Exception as e:
        print(f"DeviceManager    : ERROR ({e})")
        return False

    print("=" * 60)
    print("ENVIRONMENT HEALTH: READY FOR PHASE 2")
    print("=" * 60)
    return True


if __name__ == "__main__":
    success = run_diagnostics()
    sys.exit(0 if success else 1)
