import streamlit as st
from dotenv import load_dotenv
load_dotenv()

from agent import run_research_agent
from rag   import ingest

st.set_page_config(
    page_title="Research Assistant Agent",
    page_icon="",
    layout="wide"
)

st.title("Autonomous Research Assistant Agent")
st.caption("Powered by GPT-4o | RAG | LLM-as-Judge | Agentic Loop")

with st.sidebar:
    st.header("Agent Settings")
    st.markdown("**Primary Model:** GPT-4o")
    st.markdown("**Judge Model:** GPT-4o-mini")
    st.markdown("**RAG:** ChromaDB")
    st.markdown("**Max Retries:** 3")
    st.markdown("**Min Score:** 7/10")

    st.divider()

    st.subheader("Upload Doc to RAG")
    uploaded = st.file_uploader("Upload a .txt file", type=["txt"])
    doc_id   = st.text_input("Document ID", value="my_doc")
    if st.button("Ingest into RAG") and uploaded:
        text = uploaded.read().decode("utf-8")
        ingest(doc_id, text, {"source": uploaded.name})
        st.success(f"Ingested '{uploaded.name}' into knowledge base")

    st.divider()
    st.subheader("Example Queries")
    examples = [
        "Latest advances in quantum computing",
        "Impact of AI on healthcare",
        "Future of renewable energy",
        "Blockchain technology applications",
    ]
    for ex in examples:
        if st.button(ex, use_container_width=True):
            st.session_state["query"] = ex

query = st.text_input(
    "Enter your research query",
    value=st.session_state.get("query", ""),
    placeholder="e.g. What are the latest advances in quantum computing?"
)

run = st.button("Run Research Agent", type="primary", use_container_width=True)

if run and query:
    st.divider()

    with st.status("Agent is working...", expanded=True) as status:
        st.write("Input validated")
        st.write("Planning research sub-tasks...")
        st.write("Retrieving from knowledge base (RAG)...")
        st.write("Searching and synthesizing...")
        st.write("LLM-as-Judge evaluating...")
        result = run_research_agent(query)
        status.update(label="Research complete!", state="complete")

    if "error" in result:
        st.error(f"Error: {result['error']}")

    else:
        ev = result.get("eval", {})
        st.subheader("Evaluation Results")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Overall Score",   f"{ev.get('score','?')}/10")
        col2.metric("Accuracy",        f"{ev.get('accuracy','?')}/10")
        col3.metric("Completeness",    f"{ev.get('completeness','?')}/10")
        col4.metric("Coherence",       f"{ev.get('coherence','?')}/10")

        with st.expander("Research Plan (Sub-tasks)"):
            for i, task in enumerate(result.get("sub_tasks", []), 1):
                st.markdown(f"**{i}.** {task}")

        st.divider()

        st.subheader("Final Research Report")
        st.markdown(result["report"])

        st.divider()

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Strengths")
            for s in ev.get("strengths", []):
                st.success(s)
        with col2:
            st.subheader("Gaps")
            for g in ev.get("gaps", []):
                st.warning(g)

        st.divider()

        st.subheader("Telemetry and Monitoring")
        traces       = result.get("traces", [])
        total_tokens = sum(t.get("tokens", 0) for t in traces)
        tool_calls   = [t for t in traces if t.get("step") == "tool_loop"]
        latencies    = [t["latency_ms"] for t in traces if "latency_ms" in t]
        avg_lat      = round(sum(latencies)/len(latencies)) if latencies else 0

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Session ID",    result.get("session_id", "?"))
        col2.metric("Trace Entries", len(traces))
        col3.metric("Attempts",      len(tool_calls))
        col4.metric("Avg Latency",   f"{avg_lat} ms")

        with st.expander("Raw Trace Log"):
            for t in traces:
                st.code(str(t), language="json")