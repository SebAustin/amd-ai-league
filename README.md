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
                                                   heading context)     └ best guess
                                                                        ─► confidence < 0.5 ⇒ ""
```

* **Saved copy of a site** (`--corpus DIR`): every page is read and ranked directly — no navigation
  errors possible.
* **Answer policy**: answer only when confident, otherwise return `""` — League graders count an
  empty answer as correct on unanswerable questions and a guess as wrong
  ([ADR 0002](docs/adr/0002-abstain-below-confidence-threshold.md), supersedes ADR 0001).
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

## Results (own eval set: 30 answerable + 8 unanswerable live Requests)

Graded like the League: uppercase, whitespace and `- . · _` removed; an unanswerable case scores
only when the answer is empty.

| Configuration | Score | Answerable | Unanswerable | Max time |
|---|---|---|---|---|
| Rules only, always answer (ADR 0001) | 28/38 (74%) | 28/30 | 0/8 | < 3 s |
| Rules only, abstain < 0.5 (ADR 0002) | 36/38 (95%) | 28/30 | 8/8 | < 3 s |
| With Qwen3-8B on MI300X | _pending GPU run_ | | | |

The unanswerable cases were written alongside the threshold, so expect some optimism on unseen
questions; add fresh cases before re-tuning.

Run it: `python eval/m4/run_eval.py` (add `--llm` inside the container).

## Develop

```bash
uv venv -p 3.12 .venv && uv pip install -p .venv -e ".[dev]"
.venv/bin/pytest --cov            # 98 tests, ~88% coverage
.venv/bin/ruff check src tests eval
```

Build the image (amd64, from the repo root):

```bash
docker buildx build --platform linux/amd64 -f matches/m4/Dockerfile \
  -t <dockerhub-user>/amd-ai-league:m4-v1 .
```

## Environment knobs

| Variable | Default | Meaning |
|---|---|---|
| `M4_TIME_BUDGET` | `25` | seconds per Request |
| `M4_MIN_CONFIDENCE` | `0.5` | abstain below this; `0` = always answer |
| `M4_DISABLE_LLM` | unset | `1` = rules only |
| `M4_DISABLE_BROWSER` | unset | `1` = never launch Chromium |
| `M4_MODEL_DIR` | `/models/llm` | weights location |
| `M4_OUTPUT_DIR` | `/app/output` | output folder |
