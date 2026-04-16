# Research Assistant Agent

An autonomous AI agent that takes a research query, searches the web, retrieves from a private knowledge base, writes a report, and evaluates itself using a second LLM.

## Setup

1. Clone the repo
2. Install dependencies
   pip install openai chromadb requests beautifulsoup4 python-dotenv streamlit
3. Add your OpenAI key — create a .env file
   OPENAI_API_KEY=sk-your-key-here
4. Seed the knowledge base (run once)
   python rag.py

## How to run

Terminal mode:
   python main.py

GUI mode:
   streamlit run app_streamlit.py

## Files

- agent.py — main agent logic, tools, guardrails, evaluator, telemetry
- rag.py — ChromaDB vector store, ingest and retrieve
- main.py — entry point for terminal
- app_streamlit.py — Streamlit web UI

## Notes

- Change the query in main.py to test different topics
- Upload your own .txt files via the Streamlit sidebar to add to RAG
- Traces are saved to traces.jsonl after each run
