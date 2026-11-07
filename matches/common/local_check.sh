#!/bin/bash
# Pre-submission checks from the League brief that run without an AMD GPU (e.g. on a Mac):
#   1. the image's lower layers ARE the mandated ROCm base (layer identity, as the grader checks)
#   2. uncompressed size under 60 GiB
#   3. the container starts and app.py writes contract JSON for the Match's inputs
# The model is switched off (no GPU here), so this proves packaging, not answers; VRAM (1-48 GiB)
# can only be checked on a GPU.
#
# Usage: matches/common/local_check.sh IMAGE m2|m3|m4 [INPUT_DIR]
#   m2: INPUT_DIR holds sample images (default eval/m2/samples)
#   m3: INPUT_DIR is a corpus (default eval/m3/kit/mc3-starter-kit/mc3-corpus); runs with no
#       network and without DAC_OVERRIDE, like the grader
set -euo pipefail
IMAGE=$1
MATCH=$2
BASE=rocm/pytorch:rocm10.0_ubuntu26.04_py3.14_pytorch_release_2.13.0
LIMIT_BYTES=$((60 * 1024 * 1024 * 1024))
fail() { echo "FAIL: $*"; exit 1; }

echo "== 1. base layer identity"
base_layers=$(docker buildx imagetools inspect "$BASE" --format \
  '{{range .Image.RootFS.DiffIDs}}{{println .}}{{end}}' 2>/dev/null || true)
[ -n "$base_layers" ] || base_layers=$(docker buildx imagetools inspect "$BASE" --format \
  '{{json .Image}}' | python3 -c 'import json,sys; d=json.load(sys.stdin); d=d.get("linux/amd64", d); print("\n".join(d["rootfs"]["diff_ids"]))')
image_layers=$(docker image inspect "$IMAGE" --format '{{range .RootFS.Layers}}{{println .}}{{end}}')
n=$(printf '%s\n' "$base_layers" | grep -c .)
[ "$(printf '%s\n' "$image_layers" | head -n "$n")" = "$(printf '%s\n' "$base_layers" | grep .)" ] \
  || fail "the first $n layers are not the mandated base (squashed, or a different FROM)"
echo "OK: first $n layers match $BASE"

echo "== 2. size"
size=$(docker image inspect "$IMAGE" --format '{{.Size}}')
python3 -c "print(f'{$size / 2**30:.1f} GiB uncompressed')"
[ "$size" -lt "$LIMIT_BYTES" ] || fail "over 60 GiB"

echo "== 3. contract run (model off)"
work=$(mktemp -d)
mkdir -p "$work/output"
name="league-check-$MATCH-$$"
cleanup() { docker rm -f "$name" >/dev/null 2>&1 || true; }
trap cleanup EXIT
case "$MATCH" in
  m2)
    input=${3:-eval/m2/samples}
    docker run -d --name "$name" --platform linux/amd64 -e M2_DISABLE_LLM=1 \
      -v "$PWD/$input:/app/input:ro" -v "$work/output:/app/output" "$IMAGE" >/dev/null
    for img in "$input"/*.png "$input"/*.jpg "$input"/*.tif; do
      [ -e "$img" ] || continue
      base=$(basename "$img")
      docker exec "$name" python3 /app/app.py --input-image "/app/input/$base" 2>/dev/null
      out="$work/output/${base%.*}_output.json"
      python3 -c "import json,sys; d=json.load(open(sys.argv[1])); assert isinstance(d['text'], str); print('OK', sys.argv[2], d)" "$out" "$base" \
        || fail "no valid output for $base"
    done ;;
  m3)
    corpus=${3:-eval/m3/kit/mc3-starter-kit/mc3-corpus}
    docker run -d --name "$name" --platform linux/amd64 --network none --cap-drop DAC_OVERRIDE \
      -e M3_DISABLE_LLM=1 -v "$PWD/$corpus:/app/corpus:ro" -v "$work/output:/app/output" \
      "$IMAGE" >/dev/null
    docker exec "$name" python3 /app/app.py --index /app/corpus 2>/dev/null || fail "--index failed"
    docker exec "$name" python3 /app/app.py --corpus /app/corpus --query-id query_01 \
      --query "Which firmware version fixed ticket ORR-1847?" 2>/dev/null
    python3 -c "import json,sys; d=json.load(open(sys.argv[1])); assert isinstance(d['answer'], str) and isinstance(d['citations'], list); print('OK', d)" \
      "$work/output/query_01_output.json" || fail "no valid output for query_01" ;;
  m4)
    docker run -d --name "$name" --platform linux/amd64 -e M4_DISABLE_LLM=1 \
      -v "$work/output:/app/output" "$IMAGE" >/dev/null
    docker exec "$name" python3 /app/app.py --id q1 \
      --question "What is the price of A Light in the Attic?" --url https://books.toscrape.com/ \
      2>/dev/null
    python3 -c "import json,sys; d=json.load(open(sys.argv[1])); assert isinstance(d['answer'], str) and isinstance(d['citations'], list); print('OK', {k: d[k] for k in ('answer', 'citations')})" \
      "$work/output/q1_output.json" || fail "no valid output for q1" ;;
  *) fail "unknown match $MATCH" ;;
esac
echo "ALL CHECKS PASSED for $IMAGE (GPU-only check left: VRAM between 1 and 48 GiB)"
