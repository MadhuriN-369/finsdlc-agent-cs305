# Agentic Requirements & SDLC Advisor (with RAG)

A multi-agent system built for **CS305 — 1st Course Project Evaluation**, which
requires an Agentic-AI application with **at least two agents** that:

1. Identify and list the software requirements for a given scenario/module.
2. Identify a suitable SDLC model for that specification.

This project implements **three agents** plus an orchestrator, and grounds them
in a domain knowledge base (currently **Banking**) using **Retrieval-Augmented
Generation (RAG)**. It is a generic pipeline: it is not tied to any one scenario.

## How it works

```
User scenario (plain English)
        │
        ▼
┌─────────────────────────┐
│  RAG retrieval           │  -> embeds the scenario, finds the top-5 most
│  (rag.py)                │     similar chunks in the banking knowledge base
└───────────┬─────────────┘
            │ retrieved regulations / standards
            ▼
┌─────────────────────────┐
│  Agent 0                 │  -> judges if the scenario has enough detail
│  Clarification Agent     │  -> if not: asks up to 4 targeted questions
└───────────┬─────────────┘     (user answers, or skips and Agent 1 assumes)
            │ scenario (+ Q&A)
            ▼
┌─────────────────────────┐
│  Agent 1                 │  -> functional requirements
│  Requirements Agent      │  -> non-functional requirements
└───────────┬─────────────┘  -> assumptions
            │ structured JSON
            ▼
┌─────────────────────────┐
│  Agent 2                 │  -> fit score (0-100) for 8 SDLC models
│  SDLC Selection Agent    │  -> recommended model + justification
└─────────────────────────┘  -> project-specific workflow sketch
```

An `Orchestrator` class coordinates the agents. Each agent has its own system
prompt (its role) and its own responsibility.

**Agent 2 chooses from:** Waterfall, Iterative Waterfall, V-Shaped, Prototype,
RAD, Incremental/Iterative, Spiral, Agile.

### RAG (domain knowledge)

1. `knowledge_base/banking.md` is split into paragraph-level chunks, each
   tagged with its `##` section (e.g. "Lending and Credit Regulations").
2. Every chunk is embedded **once** with Gemini's embedding model
   (`gemini-embedding-001`) and stored in a local numpy index
   (`knowledge_base/_indices/`).
3. For each scenario, the query is embedded and compared with all chunks by
   **cosine similarity**; the top 5 chunks are injected into the prompts of
   Agents 0, 1 and 2.
4. The UI shows the retrieved chunks (source + relevance) in a collapsible panel,
   so you can see what the agents were grounded on.

No fine-tuning and no training dataset are used. The LLM is a hosted Google
Gemini model called through its REST API. No Docker, no native DLLs, and no
external vector database are needed.

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Get a Google Gemini API key (free tier available):
   https://aistudio.google.com/apikey

3. Run the app:
   ```bash
   python -m streamlit run app.py
   ```
   Paste the key into the sidebar (or set the `GEMINI_API_KEY` environment
   variable before launching).

   If the default port is blocked on your machine (Windows `WinError 10013`):
   ```bash
   python -m streamlit run app.py --server.address 127.0.0.1 --server.port 3000
   ```

4. In the sidebar, choose **Banking** under *Domain Knowledge (RAG)* and click
   **Build index** (needed once; click **Rebuild index** after editing the
   knowledge-base file).

### Command-line alternative (no UI)

```bash
# PowerShell
$env:GEMINI_API_KEY="your-key"

# one-time: build the index
python run_cli.py --domain banking --build-index ""

# run the pipeline with RAG
python run_cli.py --domain banking "A digital bank wants a loan origination module that..."

# run without RAG
python run_cli.py "A digital bank wants a loan origination module that..."
```

## Choosing a model

`DEFAULT_MODEL` in `agents.py` is `gemini-3.1-flash-lite`. You can type any
other Gemini model name in the sidebar. Google retires old model names, so if
you get a **404**, list the models your key can use:

```bash
python -c "import os,requests; r=requests.get('https://generativelanguage.googleapis.com/v1beta/models',params={'key':os.environ['GEMINI_API_KEY'],'pageSize':200}).json(); [print(m['name']) for m in r['models'] if 'generateContent' in m['supportedGenerationMethods']]"
```

## Troubleshooting

| Error | Meaning | Fix |
|---|---|---|
| `404 Not Found` (model URL) | Model name no longer exists | Pick another model name (see above) |
| `503 Service Unavailable` / `429` | Google is overloaded or rate-limiting | Click Analyse again; the app already retries automatically |
| `ReadTimeout` | Google took too long to answer | Retry; the app retries automatically (120 s timeout) |
| `streamlit.exe ... Application Control policy` | Windows blocks the launcher | Use `python -m streamlit run app.py` |
| `WinError 10013` (port) | Port blocked or reserved | Use `--server.address 127.0.0.1 --server.port 3000` |
| "index not built yet" | RAG index missing | Click **Build index** in the sidebar |

## Files

| File | Purpose |
|---|---|
| `agents.py` | Agents 0, 1, 2, the Orchestrator, and the Gemini API call (with retries) |
| `rag.py` | Local vector store: chunking, Gemini embeddings, cosine-similarity search |
| `app.py` | Streamlit UI (input → clarification → results, RAG panel) |
| `run_cli.py` | Command-line version, also used to build the index |
| `knowledge_base/banking.md` | Banking domain knowledge (lending, KYC/AML, PCI-DSS, security, core banking, risk, digital banking) |
| `requirements.txt` | Python dependencies (`streamlit`, `requests`, `numpy`) |

Add more domains by dropping a markdown file with `## Section` headings into
`knowledge_base/` (e.g. `healthcare.md`). It appears in the Domain dropdown
automatically; click **Build index** once for it.

## Limitations

- The knowledge base is small and hand-written from public summaries of
  regulations; it is not legal advice or an authoritative source.
- LLM output is non-deterministic, and no quantitative evaluation has been done.
- The system depends on an external API, so quota, overload and model
  retirements can affect it.

## Extending toward the full project

The full problem statement describes up to 13 agents (Coordinator, Stakeholder
Interaction, Requirement Extraction, Clarification, Classification,
Conflict-Detection, Compliance, Security & Privacy, Risk-Analysis, SDLC
Selection, Documentation, Validation, Human-Approval). This version implements
Clarification + Requirements + SDLC Selection with a simple orchestrator, plus
RAG. Natural next steps:

- Add a **Human-Approval** step in the UI (accept/edit/reject) before the
  requirements or SDLC choice are treated as final.
- Add a **Conflict-Detection Agent** that checks the requirements for internal
  contradictions before Agent 2 sees them.
- Add more domains (healthcare, e-commerce) and a larger knowledge base, with a
  dedicated vector database such as Qdrant.
- Add a report export (Markdown/PDF) and a quantitative evaluation of
  requirement quality with and without RAG.