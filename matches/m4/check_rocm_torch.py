"""Fail the image build unless torch is the ROCm build from the base image."""

import sys

import torch

version = torch.__version__
if not torch.version.hip or "rocm" not in version:
    sys.exit(f"torch {version} is not a ROCm build (hip={torch.version.hip}); pip replaced it")
print(f"OK: torch {version} hip {torch.version.hip}")
