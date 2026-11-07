# AMD AI League entries

Two Match entries sharing one codebase (`src/amdleague/common` holds the resident GPU model
server, deadlines and atomic output). Inference runs locally on an AMD GPU through ROCm — no API
keys, weights baked into each image. Vocabulary: [GLOSSARY.md](GLOSSARY.md); decisions:
[docs/adr](docs/adr).

| Match | Code | Image |
|---|---|---|
| 3 · Retrieval-Augmented Generation | `src/amdleague/m3`, `matches/m3` | `amd-ai-league:m3-v1` |
| 4 · Intelligent Web Information Retrieval | `src/amdleague/m4`, `matches/m4` | `amd-ai-league:m4-v1` |

# Match 3: Retrieval-Augmented Generation

Indexes a folder of mixed documents once, then answers each Question with **the value only** and
**exactly the files that answer needs** (graded as an exact set).

```
--index /app/corpus ─► defensive walk ─► parse pdf/docx/xlsx/csv/txt/log/py ─► index.json
                       (unreadable, encrypted,   png/jpg ─► Qwen3-VL-8B transcribes the text in it
                        unknown, empty: skipped)

--query "..." ─► BM25 (withdrawn/older revisions ranked down) ─► Qwen3-VL answers, or asks ONE
                 lookup of an identifier it saw (ticket id in a log) ─► citations decided by code:
                 files holding the value + the file that supplied the identifier ─► else Refusal
```

* **Refusal** — empty answer and no citations — when the value is in no readable file (sample
  Question 10: the price exists only inside the encrypted PDF).
* **Images** are read during `--index` (startup budget), not per Question
  ([ADR 0003](docs/adr/0003-match3-single-vlm-images-transcribed-at-index.md)).
* **Contract**: a Refusal is written to `/app/output/<query-id>_output.json` before any work and
  replaced atomically; the process never exits with a traceback.

Sample kit (`sh eval/m3/fetch_kit.sh`, then `python eval/m3/run_eval.py [--llm]`):

| Configuration | Score |
|---|---|
| Rules only (no GPU — images unread, no multi-hop) | 40/200 |
| With Qwen3-VL-8B on MI300X | _pending GPU run_ |

| Variable | Default | Meaning |
|---|---|---|
| `M3_TIME_BUDGET` | `25` | seconds per Question |
| `M3_INDEX_BUDGET` | `480` | seconds for `--index`, incl. waiting for the model |
| `M3_MIN_CONFIDENCE` | `0.5` | Refusal below this |
| `M3_DISABLE_LLM` | unset | `1` = rules only |

# Match 4: Intelligent Web Information Retrieval

An agent that takes a natural-language question and a website (live URL or a saved copy), navigates
it, and returns **the one specific value asked for** together with the page it came from. Inference
runs locally on an AMD GPU through ROCm — no API keys, weights baked into the image.

## How it works (Match 4)

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
| Rules only, refuse < 0.5 (ADR 0002) | 36/38 (95%) | 28/30 | 8/8 | < 3 s |
| With Qwen3-8B on MI300X | _pending GPU run_ | | | |

The unanswerable cases were written alongside the threshold, so expect some optimism on unseen
questions; add fresh cases before re-tuning.

Run it: `python eval/m4/run_eval.py` (add `--llm` inside the container).

## Develop

```bash
uv venv -p 3.12 .venv && uv pip install -p .venv -e ".[dev]"
.venv/bin/pytest --cov            # 141 tests, ~89% coverage
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
| `M4_MIN_CONFIDENCE` | `0.5` | Refusal below this; `0` = always answer |
| `M4_DISABLE_LLM` | unset | `1` = rules only |
| `M4_DISABLE_BROWSER` | unset | `1` = never launch Chromium |
| `M4_MODEL_DIR` | `/models/llm` | weights location |
| `M4_OUTPUT_DIR` | `/app/output` | output folder |
