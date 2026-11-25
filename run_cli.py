"""
run_cli.py

Quick command-line way to run the agents without Streamlit -- useful for
a fast sanity check that your Gemini API key and model choice work.

Usage:
    set GEMINI_API_KEY=AIzaSy...
    python run_cli.py "A digital bank wants a loan origination module..."

    With RAG (domain-specific):
    python run_cli.py --domain banking "A digital bank wants a loan origination module..."
"""

import argparse
import os
import sys

from agents import Orchestrator, DEFAULT_MODEL
from rag import LocalVectorStore


def main():
    parser = argparse.ArgumentParser(
        description="Agentic Requirements & SDLC Advisor (CLI)"
    )
    parser.add_argument("scenario", help="Project scenario description")
    parser.add_argument(
        "--domain", "-d",
        default=None,
        help="Domain for RAG retrieval (e.g. banking, healthcare, ecommerce). "
             "Requires the index to be built first (use --build-index).",
    )
    parser.add_argument(
        "--build-index",
        action="store_true",
        help="Build/rebuild the vector index for the specified domain and exit.",
    )
    args = parser.parse_args()

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("Set the GEMINI_API_KEY environment variable to your Google Gemini API key first.")
        sys.exit(1)

    model = os.environ.get("GEMINI_MODEL", DEFAULT_MODEL)

    # Build index if requested
    if args.build_index:
        if not args.domain:
            print("Specify a domain with --domain to build its index.")
            sys.exit(1)
        print(f"Building index for domain '{args.domain}'...")
        store = LocalVectorStore(args.domain, api_key)
        n = store.build_index(progress_callback=lambda msg: print(f"  {msg}"))
        print(f"Done — {n} chunks indexed.")
        return

    # Retrieve domain context if a domain is specified
    domain_context = ""
    if args.domain:
        if not LocalVectorStore.index_exists(args.domain):
            print(f"Index for domain '{args.domain}' not found. "
                  f"Run with --build-index first:")
            print(f'  python run_cli.py --domain {args.domain} --build-index ""')
            sys.exit(1)
        print(f"Retrieving {args.domain} domain knowledge...\n")
        store = LocalVectorStore(args.domain, api_key)
        chunks = store.query(args.scenario, top_k=5)
        domain_context = store.format_context(chunks)
        print(f"Retrieved {len(chunks)} relevant chunks.\n")

    orchestrator = Orchestrator(api_key=api_key, model=model)

    print("Running Agent 1 (Requirements Agent)...\n")
    requirements = orchestrator.requirements_agent.run(args.scenario, domain_context)
    print(requirements.as_markdown())

    print("\n" + "=" * 70 + "\n")
    print("Running Agent 2 (SDLC Agent)...\n")
    sdlc = orchestrator.sdlc_agent.run(args.scenario, requirements, domain_context)
    print(sdlc.as_markdown())


if __name__ == "__main__":
    main()