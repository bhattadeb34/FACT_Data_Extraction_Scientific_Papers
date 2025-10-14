"""Top-level CLI for fact-llm2."""
from __future__ import annotations
import argparse
import asyncio
import os
from typing import Optional

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="fact-llm2", description="Utilities for Fact-LLM workflows")
    sub = p.add_subparsers(dest="command", required=True)
    
    agent_p = sub.add_parser("agent", help="Run an agentic workflow on a file")
    agent_p.add_argument("--model", type=str, default="gemini/gemini-2.5-flash", help="Model name for text extraction")
    # FIX: Use a supported Gemini API model name with the correct prefix
    agent_p.add_argument("--vision-model", type=str, default="gemini/gemini-2.5-pro", help="Model name for figure/vision extraction")
    agent_p.add_argument("--api-key", type=str, default=os.environ.get("GEMINI_API_KEY"), help="API key")
    agent_p.add_argument("--file-path", type=str, default="./sample_papers/sample1.md", help="Path to the input file")
    agent_p.add_argument("--workflow-name", type=str, default="default", help="Workflow name")
    agent_p.add_argument("--dependency-workflow", type=str, default=None, help="Optional dependency workflow")
    agent_p.add_argument("--instruction", type=str, default=None, help="Prompt template file path")
    agent_p.add_argument("--extract-figures", action="store_true", help="Extract data from figures using vision model")
    agent_p.add_argument("--output-dir", type=str, default=None, help="Output directory (default: same as input file)")
    
    return p

def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    
    if args.command == "agent":
        from agentic_workflow import main as agent_main
        
        model = args.model
        vision_model = args.vision_model
        api_key = args.api_key or os.environ.get("GEMINI_API_KEY")
        if not api_key:
            api_key = input("Enter your API key: ").strip()
        
        asyncio.run(agent_main(
            model=model,
            api_key=api_key,
            workflow_name=args.workflow_name,
            dependency=args.dependency_workflow,
            file_path=args.file_path,
            instruction=args.instruction,
            extract_figures=args.extract_figures,
            vision_model=vision_model,
            output_dir=args.output_dir
        ))
        return 0
    
    parser.print_help()
    return 1

if __name__ == "__main__":
    raise SystemExit(main())