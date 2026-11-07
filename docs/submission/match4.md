# Match 4 submission — lablab.ai project page

Copy each block into the matching field of the lablab submission form. The bracketed `[...]` items
are yours to fill in.

## Project title

QuietAnswer: a web agent that cites its page or says nothing

## Short description (one line)

An AI agent on an AMD GPU that reads a website, returns the exact value you asked for with the
page it came from, and gives an empty answer when it can't prove it.

## Long description

QuietAnswer is our entry for AMD AI League Match 4 (Intelligent Web Information Retrieval). You
give it a question in plain language and a website. It browses the site, finds the single value
you asked for (a price, a date, an email or a part number) and returns it with the URL of the page
it came from.

It runs entirely on an AMD Instinct GPU through ROCm. The open model (Qwen3-8B) is built into the
container, so it needs no API key and no outside AI service.

How it works:

1. **Targeted browsing.** It stays on the starting site and visits the most promising links first:
   "contact" for an email, "pricing" for a price. A headless Chromium browser is used only for pages
   that build their content with JavaScript.
2. **Values stay next to their labels.** Pages are split into passages that keep their headings.
   Each table row becomes a labelled line such as `Price: £51.77`, and structured data embedded in
   the page is read too.
3. **Rank, then read.** BM25 picks the best passages. The model, kept loaded on the GPU, reads the
   value from them. Simple rules (label matching, and patterns for prices, dates and emails) back
   it up when the model is unsure.
4. **An empty answer when unsure.** League grading counts an empty answer as correct when the site
   doesn't contain the answer, and a guess as wrong. QuietAnswer therefore only answers when it is
   confident. A value it can't find on any page it read, a model reply of "not here", or a bare
   number with no label all lead to an empty answer.

Results on our own test set of 38 live questions (30 answerable, 8 not):

- Always answering: 28/38 (74%)
- Answering only when confident: **36/38 (95%)**

The retrieval and the extraction are identical in both runs. The 21-point gain comes only from
knowing when to stay quiet.

Engineering for the League rules:

- Built on the mandated `rocm/pytorch` base image. A pip constraints file pins the image's own ROCm
  build of PyTorch, and the build fails if any dependency tries to swap it for a CUDA build.
- One resident model server loads the weights onto the GPU once and serves every question over a
  Unix socket, so each question's process starts in milliseconds.
- Every run writes a valid output file before its first network request, replaces it atomically,
  stops within a hard 25-second budget, and never ends with a traceback.
- 164 automated tests with 89% coverage, plus continuous integration (ruff, pytest, hadolint).

The same codebase also powers our Match 3 entry (RAG with exact citations), which scores 200/200 on
the official sample kit running on an AMD GPU.

## Technology tags

AMD Instinct MI300X · ROCm · PyTorch · Qwen3 · Hugging Face Transformers · Docker · Python ·
httpx · BeautifulSoup · Playwright (Chromium) · BM25

## Category

Web Scraping & Data Extraction (secondary: AI Agents)

## Links

- GitHub: https://github.com/SebAustin/amd-ai-league (make the repository public before you submit)
- Docker image: `docker.io/[your-dockerhub-username]/amd-ai-league:m4-v1`
- Demo video / slides: [optional, record later]
