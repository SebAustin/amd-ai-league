*Draft LinkedIn post — tag AMD and lablab.ai when posting.*

My AI agent got 21 points better by learning to say nothing. 🤐

For the #AMDAILeague on @lablab.ai I'm building an agent that reads a website and returns the one
value you asked for — a price, a date, a part number — with the page it came from. It runs fully
on an AMD Instinct MI300X with ROCm: open model, no API keys.

The surprise: the graders count an *empty* answer as correct when the site doesn't contain the
answer, and a guess as wrong. So I stopped making the agent always answer and taught it when to
stay quiet — a value it can't find on any page it read, or a model that says "not here", now means
no answer.

Same retrieval, same model:
• Always answer → 74%
• Answer only when confident → 95%

Knowing what you don't know is a feature. Next up: a RAG system that reads PDFs, spreadsheets and
photos — and cites exactly the files it used.

#AMD #ROCm #AI #RAG #lablab
