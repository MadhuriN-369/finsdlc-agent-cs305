"""
app.py

Streamlit demo UI for the three-agent system:
  Agent 0 (Clarification Agent) -> Agent 1 (Requirements Agent) -> Agent 2 (SDLC Agent)

Run with:
    streamlit run app.py

You'll be prompted for your Hugging Face API token in the sidebar (or set
the HF_TOKEN environment variable / a .env file before launching).
"""

import os
import streamlit as st

from agents import Orchestrator, DEFAULT_MODEL

st.set_page_config(page_title="Agentic Requirements + SDLC Advisor", page_icon="🤖", layout="wide")

st.title("🤖 Agentic Requirements & SDLC Advisor")
st.caption(
    "A three-agent system: Agent 0 checks if your scenario has enough detail "
    "(and asks if not), Agent 1 extracts software requirements, Agent 2 "
    "recommends a suitable SDLC model based on those requirements."
)

# --- Sidebar: configuration -------------------------------------------------
with st.sidebar:
    st.header("Configuration")
    default_token = os.environ.get("HF_TOKEN", "")
    hf_token = st.text_input(
        "Hugging Face API token",
        value=default_token,
        type="password",
        help="Get one at https://huggingface.co/settings/tokens. "
        "Needs 'Inference' permission. Not stored anywhere.",
    )
    model_name = st.text_input(
        "Model",
        value=DEFAULT_MODEL,
        help="Any instruction-tuned chat model available on your HF Inference "
        "provider. Change this if the default is unavailable/rate-limited "
        "on your account.",
    )
    provider = st.text_input(
        "Provider (optional)",
        value="",
        help="Leave blank to let Hugging Face auto-select a provider "
        "(e.g. 'hf-inference', 'together', 'groq'). Only set this if you "
        "know your account needs a specific one.",
    )
    st.markdown("---")
    st.markdown(
        "**Agents in this system:**\n"
        "0. Clarification Agent\n"
        "1. Requirements Agent\n"
        "2. SDLC Selection Agent\n\n"
        "Coordinated by a simple orchestrator: Agent 0 gates Agent 1, whose "
        "output feeds Agent 2."
    )
    if st.button("Start over"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

# --- Session state -----------------------------------------------------------
defaults = {
    "stage": "input",       # input -> clarify -> results
    "scenario": "",
    "clarification": None,  # ClarificationOutput
    "requirements": None,   # RequirementsOutput
    "sdlc": None,            # SDLCOutput
}
for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value


def get_orchestrator():
    return Orchestrator(hf_token=hf_token, model=model_name, provider=provider or None)


def run_requirements_and_sdlc(scenario_text: str):
    orchestrator = get_orchestrator()
    with st.spinner("Agent 1 (Requirements Agent) analysing the scenario..."):
        requirements = orchestrator.requirements_agent.run(scenario_text)
    with st.spinner("Agent 2 (SDLC Agent) reasoning about SDLC fit..."):
        sdlc = orchestrator.sdlc_agent.run(scenario_text, requirements)
    st.session_state.requirements = requirements
    st.session_state.sdlc = sdlc
    st.session_state.stage = "results"


# --- Stage: input ------------------------------------------------------------
example_scenarios = {
    "-- choose an example --": "",
    "Loan origination module": (
        "A digital bank wants a loan origination module that lets customers apply "
        "for personal loans online, have their creditworthiness assessed "
        "automatically, and receive an approval or rejection decision, with all "
        "steps compliant with lending regulations and full audit trails."
    ),
    "Fraud detection service": (
        "A payments company wants a real-time fraud detection service that flags "
        "suspicious transactions as they occur, learns from analyst feedback, "
        "and integrates with the existing core banking system without downtime."
    ),
    "KYC customer onboarding": (
        "A retail bank wants to automate customer onboarding and KYC checks, "
        "verifying identity documents, screening against sanctions lists, and "
        "escalating high-risk cases to a human compliance officer."
    ),
}

if st.session_state.stage == "input":
    col1, col2 = st.columns([1, 2])
    with col1:
        chosen_example = st.selectbox("Example scenarios", list(example_scenarios.keys()))
    with col2:
        scenario = st.text_area(
            "Project scenario / module description",
            value=example_scenarios[chosen_example],
            height=120,
            placeholder="Describe the system, module, or feature in plain English...",
        )

    run_clicked = st.button("Analyse scenario", type="primary", disabled=not scenario.strip())

    if run_clicked:
        if not hf_token:
            st.error("Please enter your Hugging Face API token in the sidebar.")
            st.stop()

        st.session_state.scenario = scenario
        orchestrator = get_orchestrator()
        with st.spinner("Agent 0 (Clarification Agent) checking the scenario for gaps..."):
            clarification = orchestrator.clarification_agent.run(scenario)
        st.session_state.clarification = clarification

        if clarification.sufficient or not clarification.questions:
            run_requirements_and_sdlc(scenario)
        else:
            st.session_state.stage = "clarify"
        st.rerun()
    else:
        st.info("Enter or choose a scenario above, then click **Analyse scenario**.")

# --- Stage: clarify -----------------------------------------------------------
elif st.session_state.stage == "clarify":
    clarification = st.session_state.clarification
    st.subheader("🧐 Agent 0 — Clarification Agent")
    st.warning(
        f"This scenario looks under-specified: {clarification.reasoning}\n\n"
        "Answer what you can below (skip any you're unsure of) so Agent 1 "
        "can produce more accurate requirements instead of guessing."
    )

    with st.form("clarification_form"):
        answers = []
        for i, q in enumerate(clarification.questions):
            answers.append(st.text_input(q, key=f"clarify_answer_{i}"))
        col_a, col_b = st.columns(2)
        with col_a:
            submit_answers = st.form_submit_button("Submit answers & continue", type="primary")
        with col_b:
            skip = st.form_submit_button("Skip — proceed with assumptions instead")

    if submit_answers or skip:
        final_scenario = st.session_state.scenario
        if submit_answers:
            final_scenario = Orchestrator.merge_answers(
                st.session_state.scenario, clarification.questions, answers
            )
        run_requirements_and_sdlc(final_scenario)
        st.rerun()

    with st.expander("Raw Agent 0 response"):
        st.code(clarification.raw_text)

# --- Stage: results -----------------------------------------------------------
elif st.session_state.stage == "results":
    clarification = st.session_state.clarification
    requirements = st.session_state.requirements
    sdlc = st.session_state.sdlc

    if clarification and clarification.questions:
        st.success("Agent 0 gathered clarifications before proceeding.")
        with st.expander("What Agent 0 asked, and what you answered"):
            st.markdown(f"**Agent 0's judgment:** {clarification.reasoning}")
            for q in clarification.questions:
                st.markdown(f"- {q}")
    elif clarification:
        st.info(f"Agent 0 judged the scenario sufficient: {clarification.reasoning}")

    st.subheader("🧩 Agent 1 — Requirements Agent")
    st.markdown(requirements.as_markdown())

    st.subheader("🛠️ Agent 2 — SDLC Selection Agent")
    st.markdown(sdlc.as_markdown())

    with st.expander("Raw agent outputs (for debugging / appendix)"):
        if clarification:
            st.markdown("**Agent 0 raw response:**")
            st.code(clarification.raw_text)
        st.markdown("**Agent 1 raw response:**")
        st.code(requirements.raw_text)
        st.markdown("**Agent 2 raw response:**")
        st.code(sdlc.raw_text)

    if st.button("Analyse another scenario"):
        for key in ["stage", "scenario", "clarification", "requirements", "sdlc"]:
            st.session_state[key] = defaults[key]
        st.rerun()