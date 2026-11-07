# Match 2 — Optical Character Recognition: licence plates and road signs.
# Build from the repo root:
#   docker buildx build --platform linux/amd64 -f matches/m2/Dockerfile -t <user>/amd-ai-league:m2-v1 .
# The Harness runs: python3 /app/app.py --input-image /app/input/image_01.png
#
# The League mandates this exact ROCm PyTorch base image; do not change it or squash its layers.
FROM rocm/pytorch:rocm10.0_ubuntu26.04_py3.14_pytorch_release_2.13.0

# The same vision-language model as Match 3 (reads Latin and Chinese plates).
# ~17 GB of weights: with the 29.2 GiB base this stays well under the 60 GiB image limit.
ARG MODEL_ID=Qwen/Qwen3-VL-8B-Instruct

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    AMDLEAGUE_MODEL_DIR=/models/vlm \
    M2_OUTPUT_DIR=/app/output

WORKDIR /app

# pip can silently replace the base image's ROCm torch with a CUDA build (the brief measured
# torch 2.13.0+rocm10.0.0 -> 2.14.0+cu130 after `pip install -U transformers`). Pin the exact
# installed torch with a constraints file and verify after every install step.
COPY matches/common/rocm_guard.py /usr/local/bin/rocm_guard.py
RUN python3 /usr/local/bin/rocm_guard.py --pin > /etc/pip-torch-constraint.txt \
 && cat /etc/pip-torch-constraint.txt \
 && python3 /usr/local/bin/rocm_guard.py
ENV PIP_CONSTRAINT=/etc/pip-torch-constraint.txt

COPY matches/m2/requirements.txt /app/requirements.txt
RUN python3 -m pip install -r /app/requirements.txt && python3 /usr/local/bin/rocm_guard.py

COPY pyproject.toml README.md ./
COPY src ./src
RUN python3 -m pip install --no-deps . && python3 /usr/local/bin/rocm_guard.py

# Weights ship in the image so startup never depends on a download.
RUN HF_HUB_OFFLINE=0 python3 -c "from huggingface_hub import snapshot_download; \
snapshot_download('${MODEL_ID}', local_dir='/models/vlm', \
allow_patterns=['*.json', '*.safetensors', '*.txt', '*.model', '*.jinja', 'tokenizer*'])"

# Prove the vision processor and model config load offline from the baked weights (the
# Qwen3-VL processor also needs torchvision), and that the torch stack is still ROCm.
RUN python3 -c "from transformers import AutoConfig, AutoProcessor; \
AutoProcessor.from_pretrained('/models/vlm'); AutoConfig.from_pretrained('/models/vlm'); \
import PIL, torchvision; print('vision stack OK')" \
 && python3 /usr/local/bin/rocm_guard.py

COPY matches/m2/app.py /app/app.py
COPY matches/m2/entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh && mkdir -p /app/input /app/output

# Starts the resident model server (loads the model onto the GPU once), then stays up so the
# Harness can exec app.py for each image.
ENTRYPOINT ["/app/entrypoint.sh"]
CMD []
