#!/bin/bash
# Run the Match 2 sample eval with the real vision model inside an AMD AI Notebook (no Docker).
# Upload bundle.tgz (code + eval/m2/samples) and run: bash eval/m2/gpu_notebook_run.sh
set -euo pipefail
WORK=${WORK:-$HOME/m2}
mkdir -p "$WORK" && cd "$WORK"
tar xzf bundle.tgz 2>/dev/null

# Never let pip touch the notebook's ROCm torch stack.
python3 matches/common/rocm_guard.py --pin > /tmp/rocm-pins.txt
export PIP_CONSTRAINT=/tmp/rocm-pins.txt
python3 -m pip install -q "transformers>=4.57" accelerate pillow || true
python3 matches/common/rocm_guard.py
python3 -c "import transformers, torch; print('transformers', transformers.__version__, 'torch', torch.__version__, 'gpu', torch.cuda.get_device_name(0))"

[ -f "$HOME/models/vlm/config.json" ] || \
  python3 -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen3-VL-8B-Instruct', local_dir='$HOME/models/vlm', allow_patterns=['*.json','*.safetensors','*.txt','*.model','*.jinja','tokenizer*'])"

export PYTHONPATH="$WORK/src" AMDLEAGUE_MODEL_DIR="$HOME/models/vlm" \
       M4_MODEL_SOCKET=/tmp/m2-model.sock M4_MODEL_LOG=/tmp/m2-model.log
python3 eval/m2/run_eval.py --llm 2>/tmp/m2-eval.log | tee /tmp/m2-eval.txt
echo "--- model server log tail"; tail -5 /tmp/m2-model.log
