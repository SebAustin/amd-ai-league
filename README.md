# AMD AI League — Match 4: Intelligent Web Information Retrieval

An agent that takes a natural-language question and a website (live URL or a saved copy), navigates
it, and returns **the one specific value asked for** together with the page it came from. Inference
runs locally on an AMD GPU through ROCm — no API keys, weights baked into the image.

## How it works

```
question + url ─► best-first same-site crawl ─► passages ─► BM25 rank ─► answer
                  (links scored by question       (block-aware text,   ┌ Qwen3-8B on GPU (resident)
                   overlap + answer-type hints;    tables as labelled   ├ field rule  (Label: value)
                   Chromium fallback for JS)       rows, JSON-LD,       ├ typed rules (price/date/…)
                                                   heading context)     └ best guess (never empty)
```

* **Saved copy of a site** (`--corpus DIR`): every page is read and ranked directly — no navigation
  errors possible.
* **Answer policy**: always return the best candidate with a calibrated `confidence`
  ([ADR 0001](docs/adr/0001-always-return-best-candidate.md)).
* **Contract**: a valid `<id>_output.json` is written before any network call and replaced
  atomically; the process never exits with a traceback; a hard time budget (`M4_TIME_BUDGET`, 25 s).

## Usage

```bash
python3 /app/app.py --question "What is the price of A Light in the Attic?" \
    --url https://books.toscrape.com/ --id q1
# -> /app/output/q1_output.json
```

Accepted spellings: `--question/--query/-q/--prompt`, `--url/--start-url/--website`,
`--corpus/--pages`, `--id/--query-id/--request-id`, `--input FILE` (JSON list, `{"requests": [...]}`
or JSONL), `--output-dir`.

Output:

```json
{"id": "q1", "answer": "£51.77", "citations": ["https://books.toscrape.com/catalogue/..."],
 "sources": ["..."], "source_url": "...", "confidence": 0.75, "method": "rules"}
```

## Results (own eval set, 30 live Requests)

| Configuration | Exact | Max time |
|---|---|---|
| Rules only (no GPU) | 28/30 (93%) | < 3 s |
| With Qwen3-8B on MI300X | _pending GPU run_ | |

Run it: `python eval/m4/run_eval.py` (add `--llm` inside the container).

## Develop

```bash
uv venv -p 3.12 .venv && uv pip install -p .venv -e ".[dev]"
.venv/bin/pytest --cov            # 86 tests, ~88% coverage
.venv/bin/ruff check src tests eval
```

Build the image (amd64, from the repo root):

```bash
docker buildx build --platform linux/amd64 -f matches/m4/Dockerfile \
  --build-arg BASE_IMAGE=<mandated rocm/pytorch tag> -t <dockerhub-user>/amd-ai-league:m4-v1 .
```

## Environment knobs

| Variable | Default | Meaning |
|---|---|---|
| `M4_TIME_BUDGET` | `25` | seconds per Request |
| `M4_DISABLE_LLM` | unset | `1` = rules only |
| `M4_DISABLE_BROWSER` | unset | `1` = never launch Chromium |
| `M4_MODEL_DIR` | `/models/llm` | weights location |
| `M4_OUTPUT_DIR` | `/app/output` | output folder |
