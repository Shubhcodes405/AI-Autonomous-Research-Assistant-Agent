import os
import json
import time
import uuid
import hashlib
import statistics
from typing import List, TypedDict
from pathlib import Path
from collections import defaultdict
from openai import OpenAI
from dotenv import load_dotenv
import requests
from bs4 import BeautifulSoup

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

PRIMARY_MODEL   = "gpt-4o"
JUDGE_MODEL     = "gpt-4o-mini"
MIN_JUDGE_SCORE = 7
MAX_RETRIES     = 3
TRACE_FILE      = Path("traces.jsonl")

# cache stored in memory during the session
_cache        = {}
_tokens_used  = 0
_window_start = time.time()


def _messages_to_str(messages):
    # convert messages to plain dict so we can hash them
    serializable = []
    for m in messages:
        if isinstance(m, dict):
            serializable.append(m)
        else:
            serializable.append({
                "role":    m.role,
                "content": m.content or "",
            })
    return json.dumps(serializable, sort_keys=True)


def cache_get(messages):
    key = hashlib.sha256(_messages_to_str(messages).encode()).hexdigest()[:16]
    if key in _cache:
        print(f"   [Cache] Hit! Skipping API call.")
        return _cache[key]
    return None


def cache_set(messages, value):
    key = hashlib.sha256(_messages_to_str(messages).encode()).hexdigest()[:16]
    _cache[key] = value


def check_rate_limit(tokens):
    # if we are close to 80k tokens per minute, sleep until the window resets
    global _tokens_used, _window_start
    elapsed = time.time() - _window_start
    if elapsed >= 60:
        _tokens_used  = 0
        _window_start = time.time()
    if _tokens_used + tokens > 80_000:
        wait = 60 - elapsed
        print(f"   [TPM] Token limit reached, sleeping {wait:.1f}s")
        time.sleep(max(wait, 1))
        _tokens_used  = 0
        _window_start = time.time()
    _tokens_used += tokens


class AgentLogger:
    def __init__(self):
        self.session_id = str(uuid.uuid4())[:8]
        self.traces     = []

    def log(self, step, **data):
        entry = {"ts": time.time(), "session": self.session_id, "step": step, **data}
        self.traces.append(entry)
        with TRACE_FILE.open("a") as f:
            f.write(json.dumps(entry) + "\n")
        details = "  ".join(f"{k}={v}" for k, v in data.items())
        print(f"   [Monitor] {step}: {details}")

    def summary(self):
        total_tokens = sum(t.get("tokens", 0) for t in self.traces)
        latencies    = [t["latency_ms"] for t in self.traces if "latency_ms" in t]
        avg_latency  = round(statistics.mean(latencies)) if latencies else 0
        print(f"\n   [Monitor] Session [{self.session_id}] complete.")
        print(f"   [Monitor] {len(self.traces)} trace entries | {total_tokens:,} tokens used.")
        print(f"   [Monitor] Avg latency: {avg_latency} ms")
        print(f"   [Monitor] Traces saved to {TRACE_FILE}")


# tool schemas — same pattern as prof's tools_schema in Week 1
TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web for current information on a topic.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query string"}
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "fetch_url",
            "description": "Fetch and extract text content from a URL.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "Full URL to fetch"}
                },
                "required": ["url"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "rag_retrieve",
            "description": "Search the private knowledge base for relevant info.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Query to search"}
                },
                "required": ["query"]
            }
        }
    },
]


def web_search(query):
    print(f"   [Tool] web_search: {query}")
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
            "Accept-Encoding": "gzip, deflate",
            "Connection": "keep-alive",
        }
        url  = f"https://www.google.com/search?q={query.replace(' ', '+')}&num=5"
        resp = requests.get(url, headers=headers, timeout=10)

        soup    = BeautifulSoup(resp.text, "html.parser")
        results = []

        for g in soup.select("div.g")[:6]:
            title = g.select_one("h3")
            snip  = g.select_one("div.VwiC3b") or g.select_one("span.aCOpRe") or g.select_one("div.IsZvec")
            link  = g.select_one("a")
            if title and snip:
                results.append(
                    f"Title: {title.text.strip()}\n"
                    f"Source: {link['href'] if link else ''}\n"
                    f"Content: {snip.text.strip()}"
                )

        if results:
            print(f"   [Tool] Google returned {len(results)} results")
            return "\n\n---\n\n".join(results)
        else:
            print(f"   [Tool] Google blocked, using knowledge fallback")
            return (
                f"Web search unavailable for: '{query}'. "
                f"Use your training knowledge to answer thoroughly with "
                f"specific facts, dates, companies and technical details."
            )

    except Exception as e:
        print(f"   [Tool] Google error: {e}")
        return (
            f"Web search unavailable for: '{query}'. "
            f"Use your training knowledge to answer thoroughly with "
            f"specific facts, dates, companies and technical details."
        )


def fetch_url(url):
    print(f"   [Tool] fetch_url: {url}")
    try:
        resp = requests.get(url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
        soup = BeautifulSoup(resp.text, "html.parser")
        for tag in soup(["script", "style", "nav", "footer"]):
            tag.decompose()
        return " ".join(soup.get_text().split())[:3000]
    except Exception as e:
        return f"Fetch error: {e}"


def rag_retrieve(query):
    print(f"   [Tool] rag_retrieve: {query}")
    from rag import retrieve
    hits = retrieve(query)
    if not hits:
        return "No relevant documents found."
    parts = [f"[doc:{h['meta'].get('doc_id','?')} score:{h['score']}]\n{h['text']}"
             for h in hits]
    return "\n\n---\n\n".join(parts)


TOOL_MAP = {
    "web_search":   web_search,
    "fetch_url":    fetch_url,
    "rag_retrieve": rag_retrieve,
}


def execute_tool(name, args):
    fn = TOOL_MAP.get(name)
    if not fn:
        return f"Unknown tool: {name}"
    return fn(**args)


# list of patterns we block on input
BLOCKED_PATTERNS = ["how to make", "kill", "weapon", "hack",
                    "illegal", "drug synthesis", "exploit", "malware"]


def validate_input(query):
    print(f"   [Guardrail] Checking input...")
    for pattern in BLOCKED_PATTERNS:
        if pattern in query.lower():
            return False, f"Query blocked, restricted pattern: '{pattern}'"
    if len(query.strip()) < 5:
        return False, "Query too short."
    if len(query) > 2000:
        return False, "Query too long."
    print(f"   [Guardrail] Input OK.")
    return True, "OK"


def validate_output(report):
    print(f"   [Guardrail] Checking output...")
    prompt = f"""You are a quality-control reviewer.
Analyze this report and return JSON:
{{"passed": true or false, "issues": ["list of issues, empty if none"]}}

Only flag serious issues like completely false information or total contradictions.
Do not flag missing citations or general well known facts.

Report: \"\"\"{report[:2000]}\"\"\"
Return ONLY valid JSON."""

    response = client.chat.completions.create(
        model=JUDGE_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        response_format={"type": "json_object"},
    )
    result = json.loads(response.choices[0].message.content)
    print(f"   [Guardrail] Output passed={result.get('passed', True)}")
    return result.get("passed", True), result.get("issues", [])


def evaluate(query, report):
    # second LLM scores the report written by the first LLM
    print(f"   [Eval] LLM-as-Judge scoring...")
    prompt = f"""You are an expert research evaluator. Score this report 1-10.
Query: {query}
Report: \"\"\"{report}\"\"\"
Return ONLY valid JSON:
{{
  "score": <1-10>,
  "accuracy": <1-10>,
  "completeness": <1-10>,
  "coherence": <1-10>,
  "strengths": ["..."],
  "gaps": ["..."],
  "improvement_suggestions": "..."
}}"""
    response = client.chat.completions.create(
        model=JUDGE_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        response_format={"type": "json_object"},
    )
    result = json.loads(response.choices[0].message.content)
    print(f"   [Eval] Score: {result.get('score')}/10")
    return result


# state definition, same TypedDict pattern as Exercise 7
class ResearchState(TypedDict):
    query:       str
    sub_tasks:   List[str]
    rag_context: str
    report:      str
    eval_result: dict
    attempt:     int
    status:      str


def planner_node(state: ResearchState) -> ResearchState:
    # breaks the query into smaller sub-questions
    print(f"\n   [Planner] Analyzing query...")
    t0 = time.time()

    response = client.chat.completions.create(
        model=JUDGE_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a research planning agent. "
                    "Break the query into 3-5 specific sub-questions. "
                    "Return ONLY valid JSON like: {\"questions\": [\"q1\", \"q2\", \"q3\"]}"
                )
            },
            {"role": "user", "content": state["query"]}
        ],
        temperature=0.2,
        response_format={"type": "json_object"},
    )

    latency   = int((time.time() - t0) * 1000)
    parsed    = json.loads(response.choices[0].message.content)
    sub_tasks = list(parsed.values())[0]

    state["sub_tasks"] = sub_tasks
    print(f"   [Planner] Sub-tasks: {sub_tasks}  latency={latency}ms")
    return state


def rag_node(state: ResearchState) -> ResearchState:
    # retrieves relevant context from ChromaDB before the tool loop
    print(f"\n   [RAG] Retrieving context from knowledge base...")
    from rag import retrieve
    context = ""
    for task in state["sub_tasks"][:2]:
        hits = retrieve(task)
        if hits:
            context += f"\n[Knowledge base: {task}]\n"
            context += "\n".join(h["text"] for h in hits[:3])
    state["rag_context"] = context
    print(f"   [RAG] Retrieved {len(context)} chars.")
    return state


def tool_loop_node(state: ResearchState) -> ResearchState:
    # native tool calling loop, same pattern as prof Week 1 run_native_agent
    print(f"\n   [Agent] Starting tool loop (attempt {state['attempt']})...")

    messages = [
        {
            "role": "system",
            "content": (
                f"You are an autonomous research agent. "
                f"Use tools to research: {state['query']}\n\n"
                f"Context from knowledge base:\n{state['rag_context']}\n\n"
                f"After gathering info, write a full research report with:\n"
                f"1. Executive summary\n"
                f"2. Key findings (with sources)\n"
                f"3. Analysis\n"
                f"4. Gaps and limitations\n"
                f"5. Conclusion\n"
                f"Use Markdown formatting."
            )
        },
        {"role": "user", "content": f"Research this thoroughly: {state['query']}"}
    ]

    for _ in range(8):

        cached = cache_get(messages)
        if cached:
            state["report"] = cached
            return state

        t0       = time.time()
        response = client.chat.completions.create(
            model=PRIMARY_MODEL,
            messages=messages,
            tools=TOOL_SCHEMAS,
            tool_choice="auto"
        )
        latency = int((time.time() - t0) * 1000)
        message = response.choices[0].message

        check_rate_limit(response.usage.total_tokens)
        print(f"   [Latency] API call: {latency}ms | tokens: {response.usage.total_tokens}")

        if not message.tool_calls:
            print(f"\n   [Agent] Report generated.")
            cache_set(messages, message.content)
            state["report"] = message.content
            return state

        print(f"\n   [Agent] Calling {len(message.tool_calls)} tool(s)...")
        messages.append(message)

        for tool_call in message.tool_calls:
            func_name = tool_call.function.name
            func_args = json.loads(tool_call.function.arguments)
            print(f"   [Tool] Calling: {func_name}({func_args})")
            result = execute_tool(func_name, func_args)
            messages.append({
                "role":         "tool",
                "tool_call_id": tool_call.id,
                "content":      str(result)
            })

    state["report"] = "Research complete (max iterations reached)."
    return state


def eval_node(state: ResearchState) -> ResearchState:
    # judge scores the report, if below threshold we retry
    result = evaluate(state["query"], state["report"])
    state["eval_result"] = result
    score = result.get("score", 0)
    if score >= MIN_JUDGE_SCORE:
        state["status"] = "DONE"
    else:
        gaps = result.get("gaps", [])
        if gaps:
            state["rag_context"] += f"\n\n[Gaps to fix]: {'; '.join(gaps)}"
        state["status"] = "IN_PROGRESS"
    return state


def run_research_agent(query, extra_docs=None):
    log = AgentLogger()

    print("\n" + "="*60)
    print(f"   RESEARCH AGENT - Session [{log.session_id}]")
    print(f"   Query: {query}")
    print("="*60)

    ok, reason = validate_input(query)
    if not ok:
        print(f"   [Guardrail] BLOCKED: {reason}")
        return {"error": reason}

    if extra_docs:
        from rag import ingest
        for doc in extra_docs:
            ingest(doc["id"], doc["text"], doc.get("metadata"))

    state: ResearchState = {
        "query":       query,
        "sub_tasks":   [],
        "rag_context": "",
        "report":      "",
        "eval_result": {},
        "attempt":     0,
        "status":      "IN_PROGRESS",
    }

    t0    = time.time()
    state = planner_node(state)
    log.log("planner", sub_tasks=state["sub_tasks"],
            latency_ms=int((time.time()-t0)*1000))

    t0    = time.time()
    state = rag_node(state)
    log.log("rag", context_chars=len(state["rag_context"]),
            latency_ms=int((time.time()-t0)*1000))

    while state["attempt"] < MAX_RETRIES and state["status"] == "IN_PROGRESS":
        state["attempt"] += 1
        print(f"\n   [Agent] Attempt {state['attempt']}/{MAX_RETRIES}")

        t0    = time.time()
        state = tool_loop_node(state)
        log.log("tool_loop", attempt=state["attempt"],
                latency_ms=int((time.time()-t0)*1000))

        t0    = time.time()
        state = eval_node(state)
        log.log("eval", attempt=state["attempt"],
                score=state["eval_result"].get("score"),
                status=state["status"],
                latency_ms=int((time.time()-t0)*1000))

    passed, issues = validate_output(state["report"])
    if not passed:
        state["report"] += f"\n\n---\n*Note: {'; '.join(issues)}*"

    log.summary()

    return {
        "session_id": log.session_id,
        "query":      query,
        "report":     state["report"],
        "eval":       state["eval_result"],
        "sub_tasks":  state["sub_tasks"],
        "traces":     log.traces,
    }


def print_telemetry_report():
    print("\n" + "="*60)
    print("   TELEMETRY REPORT")
    print("="*60)

    if not TRACE_FILE.exists():
        print("   [Monitor] No traces found.")
        return

    traces   = [json.loads(l) for l in TRACE_FILE.read_text().splitlines() if l.strip()]
    sessions = defaultdict(list)
    for t in traces:
        sessions[t.get("session", "?")].append(t)

    scores    = [t["score"] for t in traces if t.get("step") == "eval" and t.get("score")]
    latencies = [t["latency_ms"] for t in traces if "latency_ms" in t]

    print(f"\n   [Monitor] Total sessions  : {len(sessions)}")
    print(f"   [Monitor] Total traces    : {len(traces)}")

    if latencies:
        print(f"   [Latency] Avg latency     : {round(statistics.mean(latencies))} ms")
        print(f"   [Latency] Max latency     : {max(latencies)} ms")

    if scores:
        print(f"   [Eval]    Scores          : {scores}")
        print(f"   [Eval]    Avg score       : {round(statistics.mean(scores), 1)}/10")
        print(f"   [Eval]    Passed (>=7)    : {sum(1 for s in scores if s >= 7)}")
        print(f"   [Eval]    Retried (<7)    : {sum(1 for s in scores if s < 7)}")

    print("\n   [Sessions] Breakdown:")
    for sid, ts in sessions.items():
        score  = next((t.get("score") for t in ts if t.get("step") == "eval"), "-")
        lats   = [t["latency_ms"] for t in ts if "latency_ms" in t]
        avg_l  = round(statistics.mean(lats)) if lats else 0
        status = "PASS" if isinstance(score, (int, float)) and score >= 7 else "RETRY"
        print(f"   Session [{sid}] | Score: {score} | Avg latency: {avg_l}ms | Status: {status}")

    Path("telemetry_summary.json").write_text(json.dumps({
        "total_sessions": len(sessions),
        "total_traces":   len(traces),
        "avg_latency_ms": round(statistics.mean(latencies)) if latencies else 0,
        "avg_score":      round(statistics.mean(scores), 1) if scores else 0,
    }, indent=2))
    print(f"\n   [Monitor] Saved to telemetry_summary.json")
    print("="*60 + "\n")