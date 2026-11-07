# The empty answer wins: building a web-answer agent on AMD GPUs

*Draft for the AMD AI League (lablab.ai) — tag @lablab.ai and @AMD when posting.*

For Match 4 of the AMD AI League I built an agent that gets a question and a website, finds the
one value asked for — a price, a date, a part number — and says which page it came from. It runs
entirely on an AMD Instinct MI300X through ROCm: no API keys, the model is baked into the
container.

## How it works

1. **Crawl with intent.** A best-first, same-site crawl follows the links whose text looks most
   like the question ("contact" for an email, "pricing" for a price), with a headless browser only
   for pages built in JavaScript.
2. **Keep the structure.** Pages become passages that remember their headings, and table rows
   become labelled lines (`Price: £51.77`), so a value is never separated from its label.
3. **Rank, then read.** BM25 picks the best passages; Qwen3 running resident on the GPU extracts
   the value, and deterministic rules (label matching, typed patterns for prices and dates) back
   it up when the model is unsure.

## The lesson: an empty answer can be the right answer

My first version always returned its best guess — under exact-match grading an empty answer and
a wrong answer both score zero, so why not try? Then I read how the League grades questions the
site cannot answer: **an empty answer is correct, and a guess is wrong.**

So the agent now answers only when it is confident: a value the model produced that appears in
no page it read, a model that says "not in these passages", or a bare number with nothing near
it to say what it counts all push confidence below the line, and the agent returns nothing.

On my own test set of 38 live questions (30 answerable, 8 not):

| Policy | Score |
|---|---|
| Always answer | 28/38 (74%) |
| Answer only when confident | **36/38 (95%)** |

Same retrieval, same extraction — 21 points better just by knowing when to stay quiet.

## Running on AMD

* Mandated `rocm/pytorch` base image; a pip constraints file pins the image's own ROCm torch so
  no dependency can silently swap it for a CUDA build, and the build fails if it ever happens.
* One model server loads the weights onto the GPU once and answers every question over a Unix
  socket, so each per-question process starts in milliseconds.

Code (open source): <repo link>
