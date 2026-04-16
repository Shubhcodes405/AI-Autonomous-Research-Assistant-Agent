import os
import chromadb
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
chroma = chromadb.PersistentClient(path="./chroma_db")
col    = chroma.get_or_create_collection("research_docs")

EMBED_MODEL = "text-embedding-3-small"
TOP_K       = 5


def _chunk(text, size=400, overlap=50):
    # split text into overlapping chunks for better retrieval
    words, chunks = text.split(), []
    i = 0
    while i < len(words):
        chunks.append(" ".join(words[i:i+size]))
        i += size - overlap
    return chunks


def _embed(texts):
    resp = client.embeddings.create(model=EMBED_MODEL, input=texts)
    return [d.embedding for d in resp.data]


def ingest(doc_id, text, metadata=None):
    chunks = _chunk(text)
    ids    = [f"{doc_id}_{i}" for i in range(len(chunks))]
    embeds = _embed(chunks)
    metas  = [{**(metadata or {}), "doc_id": doc_id, "chunk": i}
               for i in range(len(chunks))]
    col.upsert(ids=ids, embeddings=embeds, documents=chunks, metadatas=metas)
    print(f"   [RAG] Ingested '{doc_id}' -> {len(chunks)} chunks")


def retrieve(query):
    q_emb = _embed([query])[0]
    res   = col.query(query_embeddings=[q_emb], n_results=TOP_K)
    docs  = res["documents"][0]
    metas = res["metadatas"][0]
    dists = res["distances"][0]
    return [{"text": d, "meta": m, "score": round(1 - s, 3)}
            for d, m, s in zip(docs, metas, dists)]


# sample docs to seed the knowledge base
SAMPLE_DOCS = [
    {
        "id": "ai_overview",
        "text": """
        Artificial Intelligence is intelligence demonstrated by machines.
        Large Language Models like GPT-4 are transformer-based neural networks
        trained on massive text corpora. Key milestones: GPT-3 (2020),
        ChatGPT (2022), GPT-4 (2023). Applications include code generation,
        research synthesis, and autonomous agents.
        """,
        "meta": {"source": "sample", "topic": "AI"}
    },
    {
        "id": "quantum_computing",
        "text": """
        Quantum computing uses superposition and entanglement to perform
        computations. Qubits represent 0 and 1 simultaneously. IBM, Google,
        and IonQ are leading companies. Google achieved quantum supremacy in 2019.
        Applications include cryptography, drug discovery, and optimization.
        """,
        "meta": {"source": "sample", "topic": "Quantum"}
    },
    {
        "id": "climate_change",
        "text": """
        Climate change refers to long-term shifts in global temperatures caused
        by human activities. CO2 reached 421 ppm in 2023. The Paris Agreement
        targets 1.5 degrees Celsius warming limit. Renewable energy is growing
        rapidly. Extreme weather events are increasing in frequency.
        """,
        "meta": {"source": "sample", "topic": "Climate"}
    },
]

if __name__ == "__main__":
    print("="*60)
    print("   Seeding RAG knowledge base...")
    print("="*60)
    for doc in SAMPLE_DOCS:
        ingest(doc["id"], doc["text"], doc["meta"])
    print("\n   [RAG] Done! Knowledge base ready.")