"""
agents.py

Three collaborating agents for CS305 Course Project Evaluation #1:

  0. ClarificationAgent - reads the raw scenario FIRST and decides whether
                          it has enough detail to produce complete,
                          unambiguous requirements. If not, it asks up to a
                          handful of targeted questions instead of letting
                          the next agent silently guess.
  1. RequirementsAgent  - reads the (possibly clarified) scenario and
                          produces a structured list of functional and
                          non-functional software requirements.
  2. SDLCAgent          - reads the scenario + the requirements produced by
                          RequirementsAgent, reasons about project
                          characteristics (regulatory criticality, change
                          frequency, risk, etc.), and recommends a suitable
                          SDLC model with justification.

An Orchestrator class runs them in sequence: Clarification -> Requirements
-> SDLC, passing each agent's output into the next -- this is the minimal
"multi-agent" pattern: independent agents with their own role/system-prompt,
coordinated by a controller, exchanging structured messages.

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
# Agent 0: Clarification Agent
# ---------------------------------------------------------------------------

CLARIFICATION_SYSTEM_PROMPT = """You are the Clarification Agent, the first \
specialised AI agent in a multi-agent software-engineering assistant used in \
the financial sector. Your sole responsibility is to judge whether a project \
scenario has ENOUGH detail for a Requirements Agent to produce complete, \
unambiguous requirements -- BEFORE any requirements are written.

Check specifically for critical gaps such as:
- Who the users/actors/roles are
- Sensitivity/type of data involved (PII, financial, health, etc.)
- Expected scale/volume or performance expectations
- Integration points with existing systems
- Regulatory or compliance context
- Platform/deployment context (web, mobile, both, on-prem, cloud)

If the scenario is reasonably clear, say so and ask nothing -- do NOT invent
questions just to have some. Only ask about gaps that would materially
change what requirements get written. Ask at most 4 questions, each short
and specific (not generic like "please provide more details").

Respond with ONLY a single valid JSON object, no prose before or after it,
in exactly this shape:

{
  "sufficient": true or false,
  "reasoning": "<one sentence explaining your judgment>",
  "questions": ["...", "..."]
}

If "sufficient" is true, "questions" must be an empty list.
"""


@dataclass
class ClarificationOutput:
    sufficient: bool
    reasoning: str
    questions: list
    raw_text: str = field(repr=False, default="")


class ClarificationAgent:
    """Agent 0: decides if the scenario is detailed enough, or asks questions."""

    def __init__(self, client: InferenceClient, model: str = DEFAULT_MODEL):
        self.client = client
        self.model = model

    def run(self, scenario: str) -> ClarificationOutput:
        messages = [
            {"role": "system", "content": CLARIFICATION_SYSTEM_PROMPT},
            {"role": "user", "content": f"Project scenario:\n{scenario}"},
        ]
        response = self.client.chat_completion(
            messages=messages, model=self.model, max_tokens=500, temperature=0.2
        )
        text = response.choices[0].message.content
        data = _extract_json(text)

        if data is None:
            # Fail safe: if we can't parse the judgment, assume it's
            # sufficient rather than blocking the whole pipeline.
            return ClarificationOutput(
                sufficient=True,
                reasoning="Could not parse clarification judgment; proceeding anyway.",
                questions=[],
                raw_text=text,
            )

        return ClarificationOutput(
            sufficient=bool(data.get("sufficient", True)),
            reasoning=data.get("reasoning", ""),
            questions=data.get("questions", []) or [],
            raw_text=text,
        )


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

You must choose ONLY from this exact set of SDLC models (the standard set \
taught in this course) -- do not invent other names and do not default to \
Agile just because it is common:

- Waterfall: stable, well-understood requirements; heavy up-front sign-off; \
low expected change; sequential phases with no overlap.
- Iterative Waterfall: like Waterfall, but with feedback loops back to \
earlier phases allowed -- used when requirements are mostly clear but some \
rework between adjacent phases is expected.
- V-Shaped: requirements are stable AND verification/validation rigor is \
paramount (e.g. safety- or compliance-critical, every requirement must map \
to a specific test phase).
- Prototype: requirements or the user interface/interaction design are \
unclear, unfamiliar, or need to be experienced to be evaluated properly; a \
quick, rough working model is needed early (e.g. for demos, user feedback, \
a trade show/deadline, an unfamiliar product form factor) before committing \
to the full build.
- RAD (Rapid Application Development): very tight timeline, requirements \
fairly clear, heavy reuse/components, willing to trade some rigor for speed.
- Incremental/Iterative: requirements are reasonably well understood as a \
whole, but the system can and should be delivered in usable pieces/\
increments over time.
- Spiral: high technical or business risk and/or high uncertainty, large or \
expensive project, needs repeated risk-analysis cycles before committing \
further investment.
- Agile: requirements are expected to change frequently during development \
and the team needs continuous stakeholder feedback and fast iteration.

Do not force-fit Agile onto every scenario. If the strongest signal in the \
requirements is "we need a quick working model to show/test before \
committing further" (a demo, a prototype, an experimental UI/hardware \
combination, a tight deadline to demonstrate feasibility, an unfamiliar \
product area), Prototype is very likely the correct answer even in a \
financial-sector context. If strict regulatory sign-off and traceable \
testing dominate, prefer V-Shaped or Waterfall/Iterative Waterfall over \
Agile even if the domain is "modern".

Given the scenario and its requirements, you must:
1. Score how well each of the 8 models above fits this project (0-100).
2. Pick the single best-fitting model and justify the choice using the \
specific requirements you were given.
3. Sketch a short project-specific workflow: 4-6 phases with 1-2 key \
activities each, appropriate to a financial/regulated context.

Respond with ONLY a single valid JSON object, no prose before or after it, \
in exactly this shape:

{
  "scores": {"Waterfall": 0, "Iterative Waterfall": 0, "V-Shaped": 0, "Prototype": 0, "RAD": 0, "Incremental/Iterative": 0, "Spiral": 0, "Agile": 0},
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
        self.clarification_agent = ClarificationAgent(self.client, model=model)
        self.requirements_agent = RequirementsAgent(self.client, model=model)
        self.sdlc_agent = SDLCAgent(self.client, model=model)

    @staticmethod
    def merge_answers(scenario: str, questions: list, answers: list) -> str:
        """Fold clarifying Q&A pairs back into the scenario text so the
        Requirements Agent sees them as additional context."""
        qa_block = "\n".join(
            f"Q: {q}\nA: {a}" for q, a in zip(questions, answers) if a and a.strip()
        )
        if not qa_block:
            return scenario
        return f"{scenario}\n\nAdditional clarifications:\n{qa_block}"

    def run(self, scenario: str):
        """Runs the full pipeline without stopping for clarification --
        useful for the CLI / batch use. The Streamlit app instead calls the
        agents individually so it can pause and show questions in the UI."""
        requirements = self.requirements_agent.run(scenario)
        sdlc = self.sdlc_agent.run(scenario, requirements)
        return requirements, sdlc