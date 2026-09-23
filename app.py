"""
app.py

Streamlit demo UI for the two-agent system:
  Agent 1 (Requirements Agent) -> Agent 2 (SDLC Agent)

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
    "A minimal two-agent system: Agent 1 extracts software requirements from a "
    "scenario, Agent 2 recommends a suitable SDLC model based on those requirements."
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
        "1. Requirements Agent\n"
        "2. SDLC Selection Agent\n\n"
        "Coordinated by a simple orchestrator that passes Agent 1's output "
        "into Agent 2."
    )

# --- Main: scenario input ---------------------------------------------------
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

run_clicked = st.button("Run agents", type="primary", disabled=not scenario.strip())

if run_clicked:
    if not hf_token:
        st.error("Please enter your Hugging Face API token in the sidebar.")
        st.stop()

    orchestrator = Orchestrator(hf_token=hf_token, model=model_name, provider=provider or None)

    with st.spinner("Agent 1 (Requirements Agent) analysing the scenario..."):
        try:
            requirements = orchestrator.requirements_agent.run(scenario)
        except Exception as e:
            st.error(f"Requirements Agent failed: {e}")
            st.stop()

    st.subheader("🧩 Agent 1 — Requirements Agent")
    st.markdown(requirements.as_markdown())

    with st.spinner("Agent 2 (SDLC Agent) reasoning about SDLC fit..."):
        try:
            sdlc = orchestrator.sdlc_agent.run(scenario, requirements)
        except Exception as e:
            st.error(f"SDLC Agent failed: {e}")
            st.stop()

    st.subheader("🛠️ Agent 2 — SDLC Selection Agent")
    st.markdown(sdlc.as_markdown())

    with st.expander("Raw agent outputs (for debugging / appendix)"):
        st.markdown("**Agent 1 raw response:**")
        st.code(requirements.raw_text)
        st.markdown("**Agent 2 raw response:**")
        st.code(sdlc.raw_text)
else:
    st.info("Enter or choose a scenario above, then click **Run agents**.")
