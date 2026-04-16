import os
import sys
from dotenv import load_dotenv

load_dotenv()

if not os.getenv("OPENAI_API_KEY"):
    print("[Error] OPENAI_API_KEY not found in .env file.")
    sys.exit(1)

from agent import run_research_agent, print_telemetry_report

if __name__ == "__main__":
    print("="*60)
    print("   RESEARCH ASSISTANT AGENT")
    print("="*60)

    # change this query to test different topics
    query = "What are the latest advances in quantum computing?"

    # uncomment this line for interactive input instead
    # query = input("\nResearch query > ").strip()

    result = run_research_agent(query)

    if "error" in result:
        print(f"\n[Error] {result['error']}")
        sys.exit(1)

    print("\n" + "="*60)
    print("   FINAL REPORT")
    print("="*60)
    print(result["report"])

    ev = result.get("eval", {})
    print("\n" + "="*60)
    print("   EVALUATION (LLM-as-Judge)")
    print("="*60)
    print(f"   Score        : {ev.get('score', '?')}/10")
    print(f"   Accuracy     : {ev.get('accuracy', '?')}/10")
    print(f"   Completeness : {ev.get('completeness', '?')}/10")
    print(f"   Coherence    : {ev.get('coherence', '?')}/10")
    print(f"   Strengths    : {ev.get('strengths', [])}")
    print(f"   Gaps         : {ev.get('gaps', [])}")

    print_telemetry_report()