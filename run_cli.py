"""
run_cli.py

Quick command-line way to run the two agents without Streamlit -- useful for
a fast sanity check that your Hugging Face token and model choice work.

Usage:
    export HF_TOKEN=hf_xxxxxxxxxxxx
    python run_cli.py "A digital bank wants a loan origination module that..."
"""

import os
import sys

from agents import Orchestrator, DEFAULT_MODEL


def main():
    if len(sys.argv) < 2:
        print('Usage: python run_cli.py "<scenario description>"')
        sys.exit(1)

    scenario = sys.argv[1]
    token = os.environ.get("HF_TOKEN")
    if not token:
        print("Set the HF_TOKEN environment variable to your Hugging Face API token first.")
        sys.exit(1)

    model = os.environ.get("HF_MODEL", DEFAULT_MODEL)
    orchestrator = Orchestrator(hf_token=token, model=model)

    print("Running Agent 1 (Requirements Agent)...\n")
    requirements = orchestrator.requirements_agent.run(scenario)
    print(requirements.as_markdown())

    print("\n" + "=" * 70 + "\n")
    print("Running Agent 2 (SDLC Agent)...\n")
    sdlc = orchestrator.sdlc_agent.run(scenario, requirements)
    print(sdlc.as_markdown())


if __name__ == "__main__":
    main()
