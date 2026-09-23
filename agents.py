"""
agents.py

Two collaborating agents for CS305 Course Project Evaluation #1:

  1. RequirementsAgent  - reads a plain-English project scenario and produces
                          a structured list of functional and non-functional
                          software requirements.
  2. SDLCAgent          - reads the scenario + the requirements produced by
                          RequirementsAgent, reasons about project
                          characteristics (regulatory criticality, change
                          frequency, risk, etc.), and recommends a suitable
                          SDLC model with justification.

An Orchestrator class runs them in sequence, passing Agent 1's output into
Agent 2 -- this is the minimal "multi-agent" pattern: independent agents
with their own role/system-prompt, coordinated by a controller, exchanging
structured messages.

No fine-tuning and no training dataset are used. Both agents call a
Hugging Face hosted instruction-tuned model through the Inference API
(huggingface_hub.InferenceClient). This is "agentic" in the sense used by
the assignment: separate agents with distinct responsibilities, each
invoking an LLM, coordinated to produce a joint deliverable.
"""

import json
import re
from dataclasses import dataclass, field
from typing import Optional

from huggingface_hub import InferenceClient


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Any instruction-tuned chat model available on your Hugging Face Inference
# provider works here. If one model is unavailable/rate-limited on your
# account, swap this string for another (e.g. a smaller model) -- nothing
# else in the code needs to change.
DEFAULT_MODEL = "meta-llama/Llama-3.1-8B-Instruct"


def _extract_json(text: str) -> Optional[dict]:
    """Best-effort extraction of a JSON object from an LLM response.

    Models sometimes wrap JSON in prose or markdown code fences. This finds
    the first {...} block and tries to parse it; falls back to None so the
    caller can show the raw text instead of crashing.
    """
    text = text.strip()
    text = re.sub(r"^```(json)?", "", text.strip(), flags=re.IGNORECASE).strip()
    text = re.sub(r"```$", "", text.strip()).strip()
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    candidate = match.group(0) if match else text
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        return None


# ---------------------------------------------------------------------------
# Agent 1: Requirements Agent
# ---------------------------------------------------------------------------

REQUIREMENTS_SYSTEM_PROMPT = """You are the Requirements Agent, a specialised AI \
agent in a multi-agent software-engineering assistant used in the financial \
sector. Your sole responsibility is requirement elicitation and structuring.

Given a plain-English description of a project, module, or feature, you must:
1. Identify the functional requirements (what the system must do).
2. Identify the non-functional requirements (security, performance, \
compliance, availability, usability, auditability, etc. as relevant).
3. Note any assumptions you had to make because the scenario did not specify \
enough detail.

Respond with ONLY a single valid JSON object, no prose before or after it, \
in exactly this shape:

{
  "scenario_summary": "<one sentence restating the scenario>",
  "functional_requirements": ["FR-1: ...", "FR-2: ...", ...],
  "non_functional_requirements": ["NFR-1: ...", "NFR-2: ...", ...],
  "assumptions": ["...", "..."]
}
"""


@dataclass
class RequirementsOutput:
    scenario_summary: str
    functional_requirements: list
    non_functional_requirements: list
    assumptions: list
    raw_text: str = field(repr=False, default="")

    def as_markdown(self) -> str:
        lines = [f"**Scenario:** {self.scenario_summary}", "", "**Functional Requirements**"]
        lines += [f"- {r}" for r in self.functional_requirements] or ["- (none extracted)"]
        lines += ["", "**Non-Functional Requirements**"]
        lines += [f"- {r}" for r in self.non_functional_requirements] or ["- (none extracted)"]
        if self.assumptions:
            lines += ["", "**Assumptions**"]
            lines += [f"- {a}" for a in self.assumptions]
        return "\n".join(lines)


class RequirementsAgent:
    """Agent 1: turns a scenario description into structured requirements."""

    def __init__(self, client: InferenceClient, model: str = DEFAULT_MODEL):
        self.client = client
        self.model = model

    def run(self, scenario: str) -> RequirementsOutput:
        messages = [
            {"role": "system", "content": REQUIREMENTS_SYSTEM_PROMPT},
            {"role": "user", "content": f"Project scenario:\n{scenario}"},
        ]
        response = self.client.chat_completion(
            messages=messages, model=self.model, max_tokens=900, temperature=0.3
        )
        text = response.choices[0].message.content
        data = _extract_json(text)

        if data is None:
            # Graceful fallback: still return something usable for the demo
            return RequirementsOutput(
                scenario_summary=scenario[:200],
                functional_requirements=[],
                non_functional_requirements=[],
                assumptions=["Could not parse structured output from the model."],
                raw_text=text,
            )

        return RequirementsOutput(
            scenario_summary=data.get("scenario_summary", ""),
            functional_requirements=data.get("functional_requirements", []),
            non_functional_requirements=data.get("non_functional_requirements", []),
            assumptions=data.get("assumptions", []),
            raw_text=text,
        )


# ---------------------------------------------------------------------------
# Agent 2: SDLC Agent
# ---------------------------------------------------------------------------

SDLC_SYSTEM_PROMPT = """You are the SDLC Selection Agent, a specialised AI \
agent in a multi-agent software-engineering assistant used in the financial \
sector. You do NOT gather requirements yourself -- you receive already-\
extracted requirements from the Requirements Agent and decide on a suitable \
Software Development Life Cycle (SDLC) model.

Base your reasoning on standard software engineering heuristics, for example:
- Stable requirements, heavy up-front approvals -> Waterfall
- Strict verification & validation needs -> V-Model
- High technical/uncertainty risk -> Spiral
- Frequently changing requirements, need for fast iteration -> Agile
- Continuous secure deployment, strong security automation needs -> DevSecOps
- High regulation AND evolving requirements -> Agile-V-Model or Agile-DevSecOps hybrid

Given the scenario and its requirements, you must:
1. Score how well each of Waterfall, V-Model, Spiral, Agile, DevSecOps, and \
a Hybrid model fit this project (0-100).
2. Pick the best-fitting model (or hybrid) and justify the choice using the \
specific requirements you were given.
3. Sketch a short project-specific workflow: 4-6 phases with 1-2 key \
activities each, appropriate to a financial/regulated context.

Respond with ONLY a single valid JSON object, no prose before or after it, \
in exactly this shape:

{
  "scores": {"Waterfall": 0, "V-Model": 0, "Spiral": 0, "Agile": 0, "DevSecOps": 0, "Hybrid": 0},
  "recommended_model": "<name>",
  "justification": "<2-4 sentences tying the recommendation to the specific requirements>",
  "workflow": [
    {"phase": "...", "activities": ["...", "..."]},
    ...
  ]
}
"""


@dataclass
class SDLCOutput:
    scores: dict
    recommended_model: str
    justification: str
    workflow: list
    raw_text: str = field(repr=False, default="")

    def as_markdown(self) -> str:
        lines = [f"**Recommended SDLC:** {self.recommended_model}", ""]
        if self.scores:
            ranked = sorted(self.scores.items(), key=lambda kv: kv[1], reverse=True)
            lines.append("**Fit scores**")
            lines += [f"- {name}: {score}%" for name, score in ranked]
            lines.append("")
        lines.append(f"**Justification:** {self.justification}")
        if self.workflow:
            lines.append("")
            lines.append("**Suggested workflow**")
            for step in self.workflow:
                phase = step.get("phase", "")
                acts = step.get("activities", [])
                lines.append(f"- **{phase}**: " + "; ".join(acts))
        return "\n".join(lines)


class SDLCAgent:
    """Agent 2: recommends an SDLC model from a scenario + requirements."""

    def __init__(self, client: InferenceClient, model: str = DEFAULT_MODEL):
        self.client = client
        self.model = model

    def run(self, scenario: str, requirements: RequirementsOutput) -> SDLCOutput:
        req_block = requirements.as_markdown()
        messages = [
            {"role": "system", "content": SDLC_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Project scenario:\n{scenario}\n\n"
                    f"Requirements extracted by the Requirements Agent:\n{req_block}"
                ),
            },
        ]
        response = self.client.chat_completion(
            messages=messages, model=self.model, max_tokens=900, temperature=0.3
        )
        text = response.choices[0].message.content
        data = _extract_json(text)

        if data is None:
            return SDLCOutput(
                scores={},
                recommended_model="Unknown (parse error)",
                justification="Could not parse structured output from the model.",
                workflow=[],
                raw_text=text,
            )

        return SDLCOutput(
            scores=data.get("scores", {}),
            recommended_model=data.get("recommended_model", ""),
            justification=data.get("justification", ""),
            workflow=data.get("workflow", []),
            raw_text=text,
        )


# ---------------------------------------------------------------------------
# Orchestrator: coordinates the two agents
# ---------------------------------------------------------------------------

class Orchestrator:
    """Runs the two agents in sequence, passing Agent 1's output to Agent 2.

    This is the "coordinator" role described in multi-agent architectures:
    it owns execution order and hands structured context from one agent to
    the next.
    """

    def __init__(self, hf_token: str, model: str = DEFAULT_MODEL, provider: Optional[str] = None):
        client_kwargs = {"api_key": hf_token}
        if provider:
            client_kwargs["provider"] = provider
        self.client = InferenceClient(**client_kwargs)
        self.requirements_agent = RequirementsAgent(self.client, model=model)
        self.sdlc_agent = SDLCAgent(self.client, model=model)

    def run(self, scenario: str):
        requirements = self.requirements_agent.run(scenario)
        sdlc = self.sdlc_agent.run(scenario, requirements)
        return requirements, sdlc
