"""Keep the base image's ROCm torch stack in place during `docker build`.

    python3 rocm_guard.py --pin   # print pip constraints pinning the installed torch stack
    python3 rocm_guard.py         # fail the build unless torch (and torchvision) are ROCm builds

pip can silently replace ROCm torch with a CUDA build when a package depends on torch (the
Match 3 brief measured torch 2.13.0+rocm10.0.0 -> 2.14.0+cu130 after `pip install -U
transformers`, which then broke torchvision).
"""

import sys
from importlib.metadata import PackageNotFoundError, version

STACK = ("torch", "torchvision", "torchaudio", "pytorch-triton-rocm", "triton")


def pins() -> list[str]:
    out = []
    for name in STACK:
        try:
            out.append(f"{name}=={version(name)}")
        except PackageNotFoundError:
            continue
    return out


def check() -> None:
    import torch

    if not torch.version.hip or "rocm" not in torch.__version__:
        sys.exit(f"torch {torch.__version__} is not a ROCm build (hip={torch.version.hip})")
    try:
        import torchvision
    except ImportError:
        torchvision = None
    if torchvision is not None and "rocm" not in torchvision.__version__:
        sys.exit(f"torchvision {torchvision.__version__} is not a ROCm build")
    print(f"OK: torch {torch.__version__} hip {torch.version.hip}"
          + (f", torchvision {torchvision.__version__}" if torchvision else ""))


if __name__ == "__main__":
    if sys.argv[1:] == ["--pin"]:
        print("\n".join(pins()))
    else:
        check()
