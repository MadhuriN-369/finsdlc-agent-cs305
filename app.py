"""
app.py

Streamlit demo UI for the three-agent system with RAG:
  Agent 0 (Clarification Agent) -> Agent 1 (Requirements Agent) -> Agent 2 (SDLC Agent)

Optionally retrieves domain-specific knowledge (Banking, Healthcare,
E-Commerce) from a local vector store to ground the agents' responses
in real regulations, standards, and best practices.

Run with:
    streamlit run app.py
"""

import os
import streamlit as st

from agents import Orchestrator, DEFAULT_MODEL
from rag import LocalVectorStore

st.set_page_config(page_title="Agentic Requirements + SDLC Advisor", page_icon="🤖", layout="wide")

st.title("🤖 Agentic Requirements & SDLC Advisor")
st.caption(
    "A three-agent system with **RAG**: Agent 0 checks if your scenario has "
    "enough detail (and asks if not), Agent 1 extracts software requirements, "
    "Agent 2 recommends a suitable SDLC model — all grounded in domain-specific "
    "knowledge when a domain is selected."
)

# --- Sidebar: configuration -------------------------------------------------
with st.sidebar:
    st.header("Configuration")
    default_key = os.environ.get("GEMINI_API_KEY", "")
    api_key = st.text_input(
        "Google Gemini API Key",
        value=default_key,
        type="password",
        help="Get one at https://aistudio.google.com/apikey. "
        "Free tier is available. Not stored anywhere.",
    )
    model_name = st.text_input(
        "Model",
        value=DEFAULT_MODEL,
        help="Any Gemini model available on your account. "
        "gemini-3.1-flash-lite is fast and free-tier eligible.",
    )

    st.markdown("---")
    st.header("🗂️ Domain Knowledge (RAG)")

    available_domains = LocalVectorStore.list_domains()
    domain_options = ["None (general purpose)"] + [d.title() for d in available_domains]
    selected_domain_label = st.selectbox(
        "Domain",
        domain_options,
        help="Select a domain to ground the agents in domain-specific "
        "regulations, standards, and best practices via RAG.",
    )
    selected_domain = (
        None
        if selected_domain_label == "None (general purpose)"
        else selected_domain_label.lower()
    )

    if selected_domain:
        index_ready = LocalVectorStore.index_exists(selected_domain)
        if index_ready:
            st.success(f"✅ {selected_domain_label} index is ready")
        else:
            st.warning(f"⚠️ {selected_domain_label} index not built yet")

        if st.button(
            "Rebuild index" if index_ready else "Build index",
            type="primary" if not index_ready else "secondary",
            disabled=not api_key,
        ):
            store = LocalVectorStore(selected_domain, api_key)
            progress_placeholder = st.empty()
            try:
                n = store.build_index(
                    progress_callback=lambda msg: progress_placeholder.info(msg)
                )
                progress_placeholder.success(f"✅ Indexed {n} chunks for {selected_domain_label}")
            except Exception as e:
                progress_placeholder.error(f"Error building index: {e}")
        if not api_key and not index_ready:
            st.caption("Enter your API key above to build the index.")

    st.markdown("---")
    st.markdown(
        "**Agents in this system:**\n"
        "0. Clarification Agent\n"
        "1. Requirements Agent\n"
        "2. SDLC Selection Agent\n\n"
        "Coordinated by a simple orchestrator: Agent 0 gates Agent 1, whose "
        "output feeds Agent 2. When a domain is selected, relevant knowledge "
        "is retrieved and injected into each agent's context."
    )
    if st.button("Start over"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

# --- Session state -----------------------------------------------------------
defaults = {
    "stage": "input",       # input -> clarify -> results
    "scenario": "",
    "domain_context": "",
    "clarification": None,
    "requirements": None,
    "sdlc": None,
    "retrieved_chunks": [],
}
for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value


def get_orchestrator():
    return Orchestrator(api_key=api_key, model=model_name)


def retrieve_domain_context(scenario_text: str) -> tuple[str, list]:
    """Retrieve relevant domain knowledge via RAG, if a domain is selected."""
    if not selected_domain or not LocalVectorStore.index_exists(selected_domain):
        return "", []
    store = LocalVectorStore(selected_domain, api_key)
    chunks = store.query(scenario_text, top_k=5)
    context = store.format_context(chunks)
    return context, chunks


def run_requirements_and_sdlc(scenario_text: str, domain_context: str):
    orchestrator = get_orchestrator()
    with st.spinner("Agent 1 (Requirements Agent) analysing the scenario..."):
        requirements = orchestrator.requirements_agent.run(scenario_text, domain_context)
    with st.spinner("Agent 2 (SDLC Agent) reasoning about SDLC fit..."):
        sdlc = orchestrator.sdlc_agent.run(scenario_text, requirements, domain_context)
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
    "Telehealth platform": (
        "A hospital network wants a telehealth platform that lets patients book "
        "virtual consultations, have video visits with physicians, receive "
        "e-prescriptions, and access their medical records — all HIPAA-compliant."
    ),
    "E-commerce marketplace": (
        "A startup wants to build an online marketplace where vendors can list "
        "products, customers can search and purchase items with multiple payment "
        "methods, and the platform handles order fulfillment, returns, and "
        "fraud detection."
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

    if selected_domain and not LocalVectorStore.index_exists(selected_domain):
        st.warning(
            f"⚠️ The **{selected_domain.title()}** domain is selected but its index "
            "hasn't been built yet. Click **Build index** in the sidebar first, "
            "or proceed without domain knowledge."
        )

    run_clicked = st.button("Analyse scenario", type="primary", disabled=not scenario.strip())

    if run_clicked:
        if not api_key:
            st.error("Please enter your Google Gemini API key in the sidebar.")
            st.stop()

        st.session_state.scenario = scenario

        # RAG retrieval
        with st.spinner("Retrieving domain knowledge..."):
            domain_context, chunks = retrieve_domain_context(scenario)
        st.session_state.domain_context = domain_context
        st.session_state.retrieved_chunks = chunks

        orchestrator = get_orchestrator()
        with st.spinner("Agent 0 (Clarification Agent) checking the scenario for gaps..."):
            clarification = orchestrator.clarification_agent.run(scenario, domain_context)
        st.session_state.clarification = clarification

        if clarification.sufficient or not clarification.questions:
            run_requirements_and_sdlc(scenario, domain_context)
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
        run_requirements_and_sdlc(final_scenario, st.session_state.domain_context)
        st.rerun()

    with st.expander("Raw Agent 0 response"):
        st.code(clarification.raw_text)

# --- Stage: results -----------------------------------------------------------
elif st.session_state.stage == "results":
    clarification = st.session_state.clarification
    requirements = st.session_state.requirements
    sdlc = st.session_state.sdlc
    retrieved_chunks = st.session_state.get("retrieved_chunks", [])

    # Show RAG context if used
    if retrieved_chunks:
        with st.expander(
            f"📚 Retrieved domain knowledge ({len(retrieved_chunks)} chunks from "
            f"{selected_domain.title() if selected_domain else 'unknown'} knowledge base)",
            expanded=False,
        ):
            for i, chunk in enumerate(retrieved_chunks, 1):
                st.markdown(
                    f"**[{i}]** _{chunk['source']}_ — relevance: {chunk['score']:.0%}"
                )
                st.markdown(chunk["text"][:500] + ("..." if len(chunk["text"]) > 500 else ""))
                st.markdown("---")

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
        for key in ["stage", "scenario", "domain_context", "clarification",
                     "requirements", "sdlc", "retrieved_chunks"]:
            st.session_state[key] = defaults[key]
        st.rerun()