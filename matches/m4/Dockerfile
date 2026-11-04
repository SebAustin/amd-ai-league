# Match 4 — Intelligent Web Information Retrieval.
# Build from the repo root:
#   docker buildx build --platform linux/amd64 -f matches/m4/Dockerfile -t <user>/amd-ai-league:m4-v1 .
#
# The League mandates this exact ROCm PyTorch base image; do not change it or squash its layers.
FROM rocm/pytorch:rocm10.0_ubuntu26.04_py3.14_pytorch_release_2.13.0

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

# pip can silently replace the base image's ROCm torch with a CUDA/CPU wheel when a dependency
# asks for torch. Pin the exact installed torch with a constraints file and verify after every
# install step that torch is still the ROCm build.
RUN python3 -c "from importlib.metadata import version; print('torch==' + version('torch'))" \
      > /etc/pip-torch-constraint.txt \
 && cat /etc/pip-torch-constraint.txt
ENV PIP_CONSTRAINT=/etc/pip-torch-constraint.txt
COPY matches/m4/check_rocm_torch.py /usr/local/bin/check_rocm_torch.py
RUN python3 /usr/local/bin/check_rocm_torch.py

COPY matches/m4/requirements.txt /app/requirements.txt
RUN python3 -m pip install -r /app/requirements.txt && python3 /usr/local/bin/check_rocm_torch.py

COPY pyproject.toml README.md ./
COPY src ./src
RUN python3 -m pip install --no-deps . && python3 /usr/local/bin/check_rocm_torch.py

# Bake the weights into the image: no downloads at container start.
RUN HF_HUB_OFFLINE=0 python3 -c "from huggingface_hub import snapshot_download; \
snapshot_download('${MODEL_ID}', local_dir='/models/llm', \
allow_patterns=['*.json', '*.safetensors', '*.txt', '*.model', 'tokenizer*'])"

# Optional headless Chromium for JavaScript-built pages. The agent works without it.
RUN if [ "$WITH_BROWSER" = "1" ]; then \
      python3 -m pip install "playwright>=1.45" && python3 /usr/local/bin/check_rocm_torch.py && \
      (python3 -m playwright install --with-deps chromium || \
       echo "WARNING: Chromium install failed; JS rendering disabled"); \
    fi

COPY matches/m4/app.py /app/app.py
COPY matches/m4/entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh && mkdir -p /app/output

ENTRYPOINT ["/app/entrypoint.sh"]
CMD []
