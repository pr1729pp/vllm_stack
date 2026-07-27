import platform
import sys
from typing import Final

import psutil
import torch


GB: Final[int] = 1024**3


def bytes_to_gb(value: int) -> float:
    """Convert bytes to gigabytes."""
    return value / GB


def print_system_info() -> None:
    """Print operating system and Python information."""
    print("=" * 60)
    print("SYSTEM INFORMATION")
    print("=" * 60)

    print(f"Operating system : {platform.system()}")
    print(f"OS version       : {platform.mac_ver()[0] or platform.release()}")
    print(f"Machine          : {platform.machine()}")
    print(f"Processor        : {platform.processor() or 'Apple Silicon'}")
    print(f"Python version   : {sys.version.split()[0]}")
    print(f"PyTorch version  : {torch.__version__}")


def print_memory_info() -> None:
    """Print system memory information."""
    memory = psutil.virtual_memory()

    print("\n" + "=" * 60)
    print("MEMORY INFORMATION")
    print("=" * 60)

    print(f"Total memory     : {bytes_to_gb(memory.total):.2f} GB")
    print(f"Available memory : {bytes_to_gb(memory.available):.2f} GB")
    print(f"Used memory      : {bytes_to_gb(memory.used):.2f} GB")
    print(f"Memory usage     : {memory.percent:.1f}%")


def select_device() -> torch.device:
    """
    Select the best available PyTorch device.

    Priority:
    1. Apple Silicon MPS
    2. NVIDIA CUDA
    3. CPU
    """
    if torch.backends.mps.is_available():
        return torch.device("mps")

    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


def print_accelerator_info(device: torch.device) -> None:
    """Print accelerator availability and selected device."""
    print("\n" + "=" * 60)
    print("ACCELERATOR INFORMATION")
    print("=" * 60)

    print(f"MPS built        : {torch.backends.mps.is_built()}")
    print(f"MPS available    : {torch.backends.mps.is_available()}")
    print(f"CUDA available   : {torch.cuda.is_available()}")
    print(f"Selected device  : {device}")

    if device.type == "cuda":
        print(f"CUDA device      : {torch.cuda.get_device_name(0)}")
        print(
            f"CUDA memory      : "
            f"{torch.cuda.get_device_properties(0).total_memory / GB:.2f} GB"
        )

    if device.type == "mps":
        print("Apple GPU backend: Metal Performance Shaders")


def test_tensor_operation(device: torch.device) -> None:
    """Run a small matrix multiplication to verify the selected device."""
    print("\n" + "=" * 60)
    print("DEVICE TEST")
    print("=" * 60)

    try:
        matrix_a = torch.randn(1000, 1000, device=device)
        matrix_b = torch.randn(1000, 1000, device=device)

        result = torch.matmul(matrix_a, matrix_b)

        # Force execution to finish before reporting success.
        if device.type == "mps":
            torch.mps.synchronize()
        elif device.type == "cuda":
            torch.cuda.synchronize()

        print("Matrix multiplication completed successfully.")
        print(f"Input device     : {matrix_a.device}")
        print(f"Result device    : {result.device}")
        print(f"Result shape     : {tuple(result.shape)}")
        print(f"Result dtype     : {result.dtype}")

    except RuntimeError as error:
        print("Device test failed.")
        print(f"Error: {error}")
        raise


def main() -> None:
    """Run all environment and device checks."""
    print_system_info()
    print_memory_info()

    device = select_device()
    print_accelerator_info(device)
    test_tensor_operation(device)

    print("\n" + "=" * 60)
    print("ENVIRONMENT CHECK COMPLETED")
    print("=" * 60)

    if device.type == "mps":
        print("Your Mac is ready for PyTorch inference using the Apple GPU.")
    elif device.type == "cuda":
        print("Your system is ready for PyTorch inference using CUDA.")
    else:
        print("No supported GPU backend was detected. PyTorch will use the CPU.")


if __name__ == "__main__":
    main()