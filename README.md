# Agentic Requirements & SDLC Advisor

A three-agent system built for **CS305 — 1st Course Project Evaluation**,
which requires an Agentic-AI application with **at least two agents** that:
1. Identify and list the software requirements for a given scenario/module.
2. Identify a suitable SDLC for that specification.

This implements that (and adds a third agent that improves the quality of
step 1) as a small, generic pipeline — not tied to any one scenario — built
on top of the larger project vision in the problem statement.

## How it works

```
User scenario (plain English)
        │
        ▼
┌───────────────────────┐
│  Agent 0               │   -> judges if the scenario has enough detail
│  Clarification Agent   │   -> if not: asks up to 4 targeted questions
└───────────┬───────────┘   (user answers, or skips and Agent 1 assumes)
            │ (scenario, optionally enriched with Q&A)
            ▼
┌───────────────────────┐
│  Agent 1               │   -> functional requirements
│  Requirements Agent    │   -> non-functional requirements
└───────────┬───────────┘   -> assumptions (for anything still unclear)
            │ (structured JSON)
            ▼
┌───────────────────────┐
│  Agent 2               │   -> fit scores per SDLC model
│  SDLC Selection Agent  │   -> recommended model + justification
└───────────────────────┘   -> sketch of a project-specific workflow
```

An `Orchestrator` class coordinates the three agents. In the Streamlit app,
Agent 0 runs first: if the scenario is missing critical details (user
roles, data sensitivity, scale, integrations, compliance context,
deployment platform), it pauses the pipeline and shows you its questions
before Agent 1 ever runs. If the scenario is already clear, it says so and
the pipeline proceeds straight to Agent 1 → Agent 2, same as before. Each
agent has its own system prompt (its "role") and its own responsibility —
this is the minimal version of the multi-agent architecture described in
the full problem statement (which lists 13 possible agents; this implements
3 of them end to end, going beyond the "at least two" the checkpoint
requires).

Both agents call a Hugging Face **hosted** instruction-tuned model through
the Inference API — **no fine-tuning and no training dataset are used or
needed.** This matches the "retrieval-free" simplest form of the
architecture; you can extend it with RAG later for the full project.

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Get a Hugging Face API token (needs "Inference" permission):
   https://huggingface.co/settings/tokens

3. Run the app:
   ```bash
   streamlit run app.py
   ```
   Paste your token into the sidebar when the app opens in your browser.

   Or, for a quick terminal test without the UI:
   ```bash
   export HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxx
   python run_cli.py "A digital bank wants a loan origination module that..."
   ```

## Choosing a model

`DEFAULT_MODEL` in `agents.py` is set to `meta-llama/Llama-3.1-8B-Instruct`.
Hugging Face's Inference API routes requests to different backend providers
depending on your account and model availability, so if you get a "model
not found" or rate-limit error:
- Try a different model string in the sidebar (or `HF_MODEL` env var for
  the CLI) — any instruction-tuned chat model works, e.g.
  `HuggingFaceH4/zephyr-7b-beta`, `Qwen/Qwen2.5-7B-Instruct`, or a smaller
  model if you're on a free tier.
- Optionally set a specific provider (e.g. `hf-inference`, `together`,
  `groq`) in the sidebar's "Provider" field.

## Files

| File | Purpose |
|---|---|
| `agents.py` | Agent 1, Agent 2, and the Orchestrator (all core logic) |
| `app.py` | Streamlit UI |
| `run_cli.py` | Command-line alternative for quick testing |
| `requirements.txt` | Python dependencies |

## Extending toward the full project

The full problem statement describes up to 13 agents (Coordinator,
Stakeholder Interaction, Requirement Extraction, Clarification,
Classification, Conflict-Detection, Compliance, Security & Privacy,
Risk-Analysis, SDLC Selection, Documentation, Validation, Human-Approval)
plus retrieval-grounded generation (RAG) over a financial knowledge base.
This checkpoint deliberately implements a slice of that: Clarification +
Requirements + SDLC Selection, run by a simple orchestrator instead of a
full agent framework. Natural next steps for later checkpoints:
- Add a small **vector store** (e.g. Chroma) with a handful of synthetic
  "policy" documents, and have Agent 1/2 retrieve from it (RAG) instead of
  relying only on the LLM's own knowledge.
- Add a **Human-Approval** step in the UI (accept/edit/reject buttons)
  before requirements or the SDLC choice are treated as final.
- Add a **Conflict-Detection Agent** that checks Agent 1's requirements for
  internal contradictions before Agent 2 sees them.