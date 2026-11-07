#!/bin/bash
# Run the Match 3 sample eval with the real vision model inside an AMD AI Notebook (no Docker).
# Usage (from the folder holding bundle.tgz): bash gpu_notebook_run.sh
set -euo pipefail
WORK=${WORK:-$HOME/m3}
mkdir -p "$WORK" && cd "$WORK"
tar xzf bundle.tgz

# Never let pip touch the notebook's ROCm torch stack.
python3 matches/common/rocm_guard.py --pin > /tmp/rocm-pins.txt
export PIP_CONSTRAINT=/tmp/rocm-pins.txt
python3 -m pip install -q --no-deps pypdf python-docx openpyxl et_xmlfile lxml
python3 -m pip install -q "transformers>=4.57" accelerate || true
python3 matches/common/rocm_guard.py
python3 -c "import transformers, torch; print('transformers', transformers.__version__, 'torch', torch.__version__, 'gpu', torch.cuda.get_device_name(0))"

# Weights (~17 GB) and the official sample kit.
[ -f "$HOME/models/vlm/config.json" ] || \
  python3 -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen3-VL-8B-Instruct', local_dir='$HOME/models/vlm', allow_patterns=['*.json','*.safetensors','*.txt','*.model','*.jinja','tokenizer*'])"
[ -d kit/mc3-starter-kit ] || { mkdir -p kit && curl -sSL -o kit/kit.zip \
  https://storage.googleapis.com/lablab-static-eu/share/mc3-starter-kit.zip && \
  python3 -c "import zipfile; zipfile.ZipFile('kit/kit.zip').extractall('kit')" && \
  sh kit/mc3-starter-kit/setup.sh; }

export PYTHONPATH="$WORK/src" AMDLEAGUE_MODEL_DIR="$HOME/models/vlm" \
       M4_MODEL_SOCKET=/tmp/m3-model.sock M4_MODEL_LOG=/tmp/m3-model.log
python3 eval/m3/run_eval.py --llm --kit kit/mc3-starter-kit 2>/tmp/m3-eval.log | tee /tmp/m3-eval.txt
echo "--- model server log tail"; tail -5 /tmp/m3-model.log
