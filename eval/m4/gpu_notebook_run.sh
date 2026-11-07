#!/bin/bash
# Run the Match 4 eval with the real model inside an AMD AI Notebook (no Docker).
# Notebooks reach only Hugging Face and pip, not the eval websites, so record the pages first on
# a machine with internet:  python eval/m4/run_eval.py --record m4pages
# then upload bundle.tgz and m4pages.tgz next to each other and run:  bash eval/m4/gpu_notebook_run.sh
set -euo pipefail
WORK=${WORK:-$HOME/m4}
mkdir -p "$WORK" && cd "$WORK"
tar xzf bundle.tgz 2>/dev/null
tar xzf m4pages.tgz 2>/dev/null

# Never let pip touch the notebook's ROCm torch stack.
python3 matches/common/rocm_guard.py --pin > /tmp/rocm-pins.txt
export PIP_CONSTRAINT=/tmp/rocm-pins.txt
python3 -m pip install -q httpx beautifulsoup4 lxml "transformers>=4.51" accelerate || true
python3 matches/common/rocm_guard.py
python3 -c "import transformers, torch; print('transformers', transformers.__version__, 'torch', torch.__version__, 'gpu', torch.cuda.get_device_name(0))"

# Weights (~16 GB).
[ -f "$HOME/models/llm/config.json" ] || \
  python3 -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen3-8B', local_dir='$HOME/models/llm', allow_patterns=['*.json','*.safetensors','*.txt','*.model','*.jinja','tokenizer*'])"

export PYTHONPATH="$WORK/src" AMDLEAGUE_MODEL_DIR="$HOME/models/llm" \
       M4_MODEL_SOCKET=/tmp/m4-model.sock M4_MODEL_LOG=/tmp/m4-model.log
python3 eval/m4/run_eval.py --llm --replay m4pages 2>/tmp/m4-eval.log | tee /tmp/m4-eval.txt
echo "--- model server log tail"; tail -5 /tmp/m4-model.log
