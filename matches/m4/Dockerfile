# Match 4 — Intelligent Web Information Retrieval.
# Build from the repo root:
#   docker buildx build --platform linux/amd64 -f matches/m4/Dockerfile -t <user>/amd-ai-league:m4-v1 .
#
# BASE_IMAGE must be the ROCm PyTorch image the Match rules mandate. The default below is a
# placeholder until the Match 4 brief confirms the exact tag; do not squash its layers.
ARG BASE_IMAGE=rocm/pytorch:latest
FROM ${BASE_IMAGE}

ARG MODEL_ID=Qwen/Qwen3-8B
ARG WITH_BROWSER=1

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    M4_MODEL_DIR=/models/llm \
    M4_OUTPUT_DIR=/app/output \
    M4_TIME_BUDGET=25

WORKDIR /app

# Fail the build if torch is not a ROCm build (protects against a CPU/CUDA wheel sneaking in).
RUN python3 -c "import torch, sys; sys.exit(0 if torch.version.hip else 'torch is not a ROCm build')"

COPY pyproject.toml README.md ./
COPY src ./src
RUN python3 -m pip install ".[gpu]" "huggingface_hub[cli]>=0.25"

# Bake the weights into the image: no downloads at container start.
RUN HF_HUB_OFFLINE=0 python3 -c "from huggingface_hub import snapshot_download; \
snapshot_download('${MODEL_ID}', local_dir='/models/llm', \
allow_patterns=['*.json', '*.safetensors', '*.txt', '*.model', 'tokenizer*'])"

# Optional headless Chromium for JavaScript-built pages. The agent works without it.
RUN if [ "$WITH_BROWSER" = "1" ]; then \
      python3 -m pip install "playwright>=1.45" && \
      (python3 -m playwright install --with-deps chromium || \
       echo "WARNING: Chromium install failed; JS rendering disabled"); \
    fi

COPY matches/m4/app.py /app/app.py
COPY matches/m4/entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh && mkdir -p /app/output

ENTRYPOINT ["/app/entrypoint.sh"]
CMD []
