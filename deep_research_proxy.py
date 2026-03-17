#!/usr/bin/env python3
"""
Deep Research Proxy (MiroFlow) for Open WebUI.

An OpenAI-compatible proxy that implements a MiroFlow-inspired agentic deep research
loop using Mistral's native function calling. When a user asks a question, the proxy
orchestrates multi-turn reasoning with tool use (SearXNG search, web page reading,
Python execution) and streams the entire research process as <think> tags to Open WebUI,
followed by a polished final answer.

Architecture:
  - Receives OpenAI-compatible chat/completions requests from Open WebUI
  - Sends requests to Mistral API with native `tools` parameter
  - Parses tool_calls from the response, executes tools, feeds results back
  - All reasoning and tool interactions streamed as <think> content
  - Final answer streamed as main content after </think>
  - Utility requests (title/tag generation) bypass the agent loop
"""

import asyncio
import html
import json
import logging
import logging.handlers
import os
import re
import subprocess
import sys
import tempfile
import time
import traceback
import uuid
from datetime import datetime, timezone
from typing import AsyncGenerator, Optional

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, JSONResponse

# --- Logging ---
LOG_DIR = "/opt/deep_research_logs"
os.makedirs(LOG_DIR, exist_ok=True)

logging.root.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
logging.root.addHandler(console_handler)

file_handler = logging.handlers.RotatingFileHandler(
    os.path.join(LOG_DIR, "proxy.log"),
    maxBytes=10 * 1024 * 1024,
    backupCount=5,
)
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
logging.root.addHandler(file_handler)

log = logging.getLogger("deep-research")

# --- Configuration ---
UPSTREAM_BASE = os.getenv("UPSTREAM_BASE", "https://api.mistral.ai/v1")
UPSTREAM_KEY = os.getenv("UPSTREAM_KEY", "4ecwQOWEBgZQP6sDNr5uZMM7EuAvTdXE")
UPSTREAM_MODEL = os.getenv("UPSTREAM_MODEL", "mistral-large-latest")
SEARXNG_URL = os.getenv("SEARXNG_URL", "http://localhost:8888")
LISTEN_PORT = int(os.getenv("DEEP_RESEARCH_PORT", "9200"))
MAX_AGENT_TURNS = int(os.getenv("MAX_AGENT_TURNS", "15"))
WEBPAGE_MAX_CHARS = 15000
PYTHON_TIMEOUT = 30
PYTHON_OUTPUT_MAX = 5000

log.info(f"Config: model={UPSTREAM_MODEL}, upstream={UPSTREAM_BASE}, searxng={SEARXNG_URL}, port={LISTEN_PORT}, max_turns={MAX_AGENT_TURNS}")

# --- Native Tool Definitions (OpenAI function-calling format) ---
NATIVE_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "searxng_search",
            "description": "Search the web using SearXNG. Returns top results with titles, URLs, and snippets. Use this to find information, verify facts, discover sources.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The search query"}
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "fetch_webpage",
            "description": "Fetch a webpage and extract its readable text content. Use this to read articles, documentation, or any web page found via search.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "The URL to fetch"},
                    "extract_info": {"type": "string", "description": "Optional: specific information to look for in the page"},
                },
                "required": ["url"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "python_exec",
            "description": "Execute Python code for calculations, data processing, or analysis. Code runs in a sandboxed subprocess with a 30-second timeout.",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {"type": "string", "description": "Python code to execute. Use print() to output results."}
                },
                "required": ["code"],
            },
        },
    },
]

# --- System Prompt ---
SYSTEM_PROMPT_TEMPLATE = """You are a deep research agent. Today is: {date}

Your goal is to thoroughly research the user's question using the available tools and provide a comprehensive, well-sourced answer.

**MANDATORY Research Process:**
You MUST perform substantial research before answering. Follow this process:
1. Start by searching for the topic with searxng_search (at least 2-3 different search queries from different angles)
2. Read the most promising sources with fetch_webpage (at least 2-3 pages)
3. If you find conflicting information, search again to verify
4. Use python_exec for any calculations or data processing
5. Only after gathering substantial information, provide your final answer

**Critical Rules:**
- You MUST use tools before answering. Do NOT answer from your training data alone.
- Perform AT LEAST 2 searches and read AT LEAST 2 web pages before giving your final answer.
- Think step by step. After each tool result, explain what you learned and what you still need to find.
- Be factual — cite your sources with URLs
- When research is complete, respond with your final answer (no tool call)
- Do NOT repeat the same search query or fetch the same URL twice
- If a tool call fails, try a different approach
"""

# --- Utility Detection (same patterns as thinking_proxy) ---
UTILITY_PATTERNS = [
    "generate a concise",
    "generate 1-3 broad tags",
    "generate a title",
    "### task:\ngenerate",
    "create a concise title",
    "generate a search query",
    "autocomplete",
]

app = FastAPI(title="Deep Research Proxy (MiroFlow)")
active_requests: dict[str, dict] = {}


def is_utility_request(messages: list[dict]) -> bool:
    """Detect automated utility requests from Open WebUI."""
    for msg in messages:
        content = msg.get("content", "")
        if isinstance(content, str):
            content_lower = content.lower()
            for pattern in UTILITY_PATTERNS:
                if pattern in content_lower:
                    return True
    return False


# ============================================================================
# Tool Execution
# ============================================================================

async def tool_searxng_search(query: str) -> str:
    """Execute a SearXNG search and return formatted results."""
    try:
        async with httpx.AsyncClient(timeout=20.0) as client:
            resp = await client.get(
                f"{SEARXNG_URL}/search",
                params={"q": query, "format": "json", "categories": "general"},
            )
            if resp.status_code != 200:
                return f"Search error: HTTP {resp.status_code}"

            data = resp.json()
            results = data.get("results", [])[:10]

            if not results:
                return "No results found."

            formatted = []
            for i, r in enumerate(results, 1):
                title = r.get("title", "No title")
                url = r.get("url", "")
                snippet = r.get("content", "")[:300]
                formatted.append(f"{i}. **{title}**\n   URL: {url}\n   {snippet}")

            return "\n\n".join(formatted)

    except Exception as e:
        return f"Search error: {str(e)}"


async def tool_fetch_webpage(url: str, extract_info: str = "") -> str:
    """Fetch a webpage and extract readable text."""
    try:
        async with httpx.AsyncClient(
            timeout=20.0,
            follow_redirects=True,
            headers={"User-Agent": "Mozilla/5.0 (compatible; DeepResearchBot/1.0)"},
        ) as client:
            resp = await client.get(url)
            if resp.status_code != 200:
                return f"Fetch error: HTTP {resp.status_code}"

            content_type = resp.headers.get("content-type", "")
            if "text/html" not in content_type and "text/plain" not in content_type:
                return f"Non-text content type: {content_type}"

            raw_html = resp.text

            # Simple but effective HTML to text extraction
            text = re.sub(r'<script[^>]*>.*?</script>', '', raw_html, flags=re.DOTALL | re.IGNORECASE)
            text = re.sub(r'<style[^>]*>.*?</style>', '', raw_html, flags=re.DOTALL | re.IGNORECASE)
            text = re.sub(r'<[^>]+>', ' ', text)
            text = html.unescape(text)
            text = re.sub(r'\s+', ' ', text).strip()
            text = re.sub(r'\n{3,}', '\n\n', text)

            if len(text) > WEBPAGE_MAX_CHARS:
                text = text[:WEBPAGE_MAX_CHARS] + "\n\n[... content truncated ...]"

            if not text.strip():
                return "Page returned no readable text content."

            result = f"**Content from {url}:**\n\n{text}"
            if extract_info:
                result = f"**Looking for: {extract_info}**\n\n{result}"
            return result

    except httpx.ReadTimeout:
        return f"Fetch error: Timeout reading {url}"
    except Exception as e:
        return f"Fetch error: {str(e)}"


def tool_python_exec(code: str) -> str:
    """Execute Python code in a sandboxed subprocess."""
    try:
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            timeout=PYTHON_TIMEOUT,
            cwd=tempfile.gettempdir(),
        )
        output = ""
        if result.stdout:
            output += result.stdout
        if result.stderr:
            output += f"\n[stderr]: {result.stderr}"
        if not output.strip():
            output = "(no output)"
        if len(output) > PYTHON_OUTPUT_MAX:
            output = output[:PYTHON_OUTPUT_MAX] + "\n[... output truncated ...]"
        return output
    except subprocess.TimeoutExpired:
        return f"Error: Code execution timed out after {PYTHON_TIMEOUT}s"
    except Exception as e:
        return f"Error executing code: {str(e)}"


async def execute_tool(tool_name: str, arguments: dict) -> str:
    """Route and execute a tool call."""
    if tool_name == "searxng_search":
        return await tool_searxng_search(arguments.get("query", ""))
    elif tool_name == "fetch_webpage":
        return await tool_fetch_webpage(
            arguments.get("url", ""),
            arguments.get("extract_info", ""),
        )
    elif tool_name == "python_exec":
        return tool_python_exec(arguments.get("code", ""))
    else:
        return f"Unknown tool: {tool_name}"


# ============================================================================
# LLM Communication (Native Function Calling)
# ============================================================================

async def call_llm(messages: list[dict], req_id: str, turn: int, include_tools: bool = True) -> dict:
    """Call the upstream LLM with native function calling. Returns the full message dict."""
    body = {
        "model": UPSTREAM_MODEL,
        "messages": messages,
        "max_tokens": 4096,
        "temperature": 0.3,
        "stream": False,
    }
    if include_tools:
        body["tools"] = NATIVE_TOOLS
        body["tool_choice"] = "auto"

    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(300.0, connect=30.0)) as client:
            resp = await client.post(
                f"{UPSTREAM_BASE}/chat/completions",
                json=body,
                headers={
                    "Authorization": f"Bearer {UPSTREAM_KEY}",
                    "Content-Type": "application/json",
                },
            )

            if resp.status_code != 200:
                error_text = resp.text[:500]
                log.error(f"[{req_id}] Turn {turn}: LLM error {resp.status_code}: {error_text}")
                return {"error": f"[LLM Error: HTTP {resp.status_code}] {error_text}"}

            data = resp.json()
            choices = data.get("choices", [])
            if not choices:
                return {"error": "[LLM Error: No choices in response]"}

            message = choices[0].get("message", {})
            finish_reason = choices[0].get("finish_reason", "")

            return {
                "message": message,
                "content": message.get("content", "") or "",
                "tool_calls": message.get("tool_calls", None),
                "finish_reason": finish_reason,
            }

    except httpx.ReadTimeout:
        return {"error": "[LLM Error: Request timed out]"}
    except Exception as e:
        return {"error": f"[LLM Error: {str(e)}]"}


async def call_llm_with_keepalive(
    messages: list[dict], req_id: str, turn: int, keepalive_queue: asyncio.Queue, include_tools: bool = True
) -> dict:
    """Call LLM while sending keepalive signals so the SSE stream doesn't stall."""
    result_holder = {"value": None, "done": False}

    async def _do_call():
        result_holder["value"] = await call_llm(messages, req_id, turn, include_tools)
        result_holder["done"] = True

    async def _keepalive():
        while not result_holder["done"]:
            await asyncio.sleep(8)
            if not result_holder["done"]:
                await keepalive_queue.put(".")

    await asyncio.gather(_do_call(), _keepalive())
    return result_holder["value"]


# ============================================================================
# Agent Loop + Streaming
# ============================================================================

async def run_deep_research(
    user_messages: list[dict],
    original_body: dict,
    req_id: str,
) -> AsyncGenerator[str, None]:
    """
    Run the deep research agent loop and stream results as SSE.
    Uses native function calling — no XML parsing needed.
    All agent reasoning goes inside <think>, final answer comes after </think>.
    """
    model_id = original_body.get("model", "miroflow")
    request_id = f"chatcmpl-dr-{uuid.uuid4().hex[:12]}"
    created = int(time.time())
    start_time = time.monotonic()

    def make_chunk(content: str, finish_reason: Optional[str] = None) -> str:
        delta = {}
        if content:
            delta["content"] = content
        data = {
            "id": request_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model_id,
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
        }
        return f"data: {json.dumps(data)}\n\n"

    # Build system prompt
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(date=today)

    # Build message history for the agent
    agent_messages = [{"role": "system", "content": system_prompt}]
    for msg in user_messages:
        if msg.get("role") != "system":
            agent_messages.append(msg)

    log.info(f"[{req_id}] Starting deep research loop, user messages: {len(user_messages)}")

    # Open the thinking block
    yield make_chunk("<think>\n")

    used_queries = set()
    keepalive_q = asyncio.Queue()
    consecutive_errors = 0
    MAX_CONSECUTIVE_ERRORS = 2
    total_tool_calls = 0  # Track research depth
    MIN_TOOL_CALLS_BEFORE_ANSWER = 3  # Must do at least this many tool calls before answering

    async def llm_with_dots(msgs, turn_num, include_tools=True):
        return await call_llm_with_keepalive(msgs, req_id, turn_num, keepalive_q, include_tools)

    def drain_keepalive():
        dots = ""
        while not keepalive_q.empty():
            try:
                dots += keepalive_q.get_nowait()
            except asyncio.QueueEmpty:
                break
        return dots

    try:
        for turn in range(1, MAX_AGENT_TURNS + 1):
            elapsed = time.monotonic() - start_time
            log.info(f"[{req_id}] Turn {turn}/{MAX_AGENT_TURNS} ({elapsed:.1f}s elapsed)")

            # Stream turn header
            yield make_chunk(f"\n**[Turn {turn}/{MAX_AGENT_TURNS}]** ")

            # Call LLM with native tools
            active_requests[req_id]["current_turn"] = turn
            result = await llm_with_dots(agent_messages, turn)

            # Emit keepalive dots
            dots = drain_keepalive()
            if dots:
                yield make_chunk(dots)

            # Check for errors
            if "error" in result:
                consecutive_errors += 1
                yield make_chunk(f"⚠️ {result['error']}\n")

                if consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
                    log.error(f"[{req_id}] Circuit breaker: {consecutive_errors} consecutive errors, aborting")
                    yield make_chunk(f"\n🛑 Aborting: {consecutive_errors} consecutive LLM errors.\n")
                    yield make_chunk("\n</think>\n\n")
                    yield make_chunk(f"**Research failed:** The upstream LLM returned repeated errors.\n\nLast error: {result['error']}")
                    yield make_chunk("", finish_reason="stop")
                    yield "data: [DONE]\n\n"
                    active_requests.pop(req_id, None)
                    return

                agent_messages.append({"role": "assistant", "content": result["error"]})
                agent_messages.append({"role": "user", "content": "There was an error. Please try a different approach."})
                continue

            # Reset error counter
            consecutive_errors = 0

            content = result["content"]
            tool_calls = result.get("tool_calls")

            # Stream full reasoning content from the model (don't truncate — this is the thinking trace)
            if content:
                yield make_chunk(f"{content}\n")

            # No tool calls = model wants to give final answer
            if not tool_calls:
                # Push back if insufficient research
                if total_tool_calls < MIN_TOOL_CALLS_BEFORE_ANSWER and turn < MAX_AGENT_TURNS - 1:
                    log.info(f"[{req_id}] Turn {turn}: Model tried to answer early ({total_tool_calls} tool calls < {MIN_TOOL_CALLS_BEFORE_ANSWER} minimum), pushing back")
                    yield make_chunk(f"\n⚠️ Only {total_tool_calls} tool calls so far — need more research. Pushing agent to continue...\n")
                    agent_messages.append({"role": "assistant", "content": content})
                    agent_messages.append({"role": "user", "content": "You haven't done enough research yet. You need to search for more information and read more sources before answering. Please use the search and fetch tools to gather more data. Do NOT answer yet."})
                    continue

                log.info(f"[{req_id}] Turn {turn}: Final answer after {total_tool_calls} tool calls")
                yield make_chunk(f"\n✅ Research complete ({total_tool_calls} tool calls across {turn} turns). Generating final answer...\n")
                yield make_chunk("\n</think>\n\n")

                # Stream the final answer in chunks
                answer = content if content else "(No answer generated)"
                for i in range(0, len(answer), 200):
                    yield make_chunk(answer[i:i + 200])

                yield make_chunk("", finish_reason="stop")
                yield "data: [DONE]\n\n"
                active_requests.pop(req_id, None)
                return

            # Process tool calls (Mistral can return multiple, we handle all)
            # Build the assistant message with tool_calls for the message history
            assistant_msg = {"role": "assistant", "content": content or None, "tool_calls": tool_calls}
            agent_messages.append(assistant_msg)

            for tc in tool_calls:
                tc_id = tc.get("id", f"call_{uuid.uuid4().hex[:8]}")
                func = tc.get("function", {})
                tool_name = func.get("name", "unknown")
                arguments_str = func.get("arguments", "{}")

                # Parse arguments
                try:
                    arguments = json.loads(arguments_str) if isinstance(arguments_str, str) else arguments_str
                except json.JSONDecodeError:
                    arguments = {}

                # Duplicate check
                query_key = f"{tool_name}:{json.dumps(arguments, sort_keys=True)}"
                if query_key in used_queries:
                    log.warning(f"[{req_id}] Turn {turn}: Duplicate tool call: {tool_name}")
                    yield make_chunk(f"⚠️ Skipping duplicate {tool_name} call.\n")
                    agent_messages.append({
                        "role": "tool",
                        "tool_call_id": tc_id,
                        "content": "Duplicate call skipped. Please use previously gathered information or try a different query.",
                    })
                    continue

                used_queries.add(query_key)

                # Stream tool call info
                if tool_name == "searxng_search":
                    yield make_chunk(f"🔍 Searching: `{arguments.get('query', '')}`\n")
                elif tool_name == "fetch_webpage":
                    yield make_chunk(f"📄 Reading: `{arguments.get('url', '')[:80]}`\n")
                elif tool_name == "python_exec":
                    code_preview = arguments.get('code', '')[:100].replace('\n', ' ')
                    yield make_chunk(f"🐍 Running code: `{code_preview}`\n")
                else:
                    yield make_chunk(f"🔧 Calling: {tool_name}\n")

                # Execute tool
                total_tool_calls += 1
                tool_start = time.monotonic()
                tool_result = await execute_tool(tool_name, arguments)
                tool_duration = time.monotonic() - tool_start

                log.info(f"[{req_id}] Turn {turn}: Tool {tool_name} completed in {tool_duration:.1f}s, result length: {len(tool_result)}")

                # Stream tool result into thinking trace (show enough to be useful)
                result_display = tool_result[:2000]
                if len(tool_result) > 2000:
                    result_display += f"\n[... {len(tool_result) - 2000} more chars truncated ...]" 
                yield make_chunk(f"\n**Result** ({tool_duration:.1f}s, {len(tool_result)} chars):\n{result_display}\n")

                # Feed tool result back using proper tool role message
                agent_messages.append({
                    "role": "tool",
                    "tool_call_id": tc_id,
                    "content": tool_result,
                })

        # Max turns reached — force answer without tools
        log.info(f"[{req_id}] Max turns ({MAX_AGENT_TURNS}) reached, forcing final answer")
        yield make_chunk(f"\n⏰ Max research turns reached. Generating answer...\n")

        agent_messages.append({
            "role": "user",
            "content": "You have reached the maximum number of research turns. Based on ALL the information gathered so far, provide your final comprehensive answer. Do NOT call any tools."
        })
        final_result = await llm_with_dots(agent_messages, MAX_AGENT_TURNS + 1, include_tools=False)
        dots = drain_keepalive()
        if dots:
            yield make_chunk(dots)

        yield make_chunk("\n</think>\n\n")
        final_answer = final_result.get("content", "") if "error" not in final_result else final_result["error"]
        for i in range(0, len(final_answer), 200):
            yield make_chunk(final_answer[i:i + 200])
        yield make_chunk("", finish_reason="stop")
        yield "data: [DONE]\n\n"

    except Exception as e:
        elapsed = time.monotonic() - start_time
        tb = traceback.format_exc()
        log.error(f"[{req_id}] Agent loop error after {elapsed:.2f}s: {e}\n{tb}")
        yield make_chunk(f"\n⚠️ Error: {str(e)}\n")
        yield make_chunk("\n</think>\n\n")
        yield make_chunk(f"**Deep Research Error**\n\nAn error occurred during research: {str(e)}")
        yield make_chunk("", finish_reason="stop")
        yield "data: [DONE]\n\n"

    finally:
        active_requests.pop(req_id, None)


# ============================================================================
# Passthrough for Utility Requests
# ============================================================================

async def stream_passthrough(
    messages: list[dict],
    original_body: dict,
    req_id: str,
) -> AsyncGenerator[str, None]:
    """Pass utility requests straight through to the upstream LLM without the agent loop."""
    model_id = original_body.get("model", "miroflow")
    request_id = f"chatcmpl-pass-{uuid.uuid4().hex[:12]}"
    created = int(time.time())

    upstream_body = {
        **original_body,
        "model": UPSTREAM_MODEL,
        "messages": messages,
        "stream": True,
    }
    for key in ("user", "chat_id", "tools", "tool_choice", "functions", "function_call"):
        upstream_body.pop(key, None)

    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(120.0, connect=30.0)) as client:
            async with client.stream(
                "POST",
                f"{UPSTREAM_BASE}/chat/completions",
                json=upstream_body,
                headers={
                    "Authorization": f"Bearer {UPSTREAM_KEY}",
                    "Content-Type": "application/json",
                },
            ) as resp:
                if resp.status_code != 200:
                    error_body = await resp.aread()
                    error_text = error_body.decode("utf-8", errors="replace")[:500]
                    chunk = {
                        "id": request_id, "object": "chat.completion.chunk",
                        "created": created, "model": model_id,
                        "choices": [{"index": 0, "delta": {"content": f"Error: {error_text}"}, "finish_reason": "stop"}],
                    }
                    yield f"data: {json.dumps(chunk)}\n\n"
                    yield "data: [DONE]\n\n"
                    return

                async for line in resp.aiter_lines():
                    if line.startswith("data: "):
                        payload = line[6:].strip()
                        if payload == "[DONE]":
                            yield "data: [DONE]\n\n"
                            return
                        try:
                            chunk = json.loads(payload)
                            chunk["model"] = model_id
                            yield f"data: {json.dumps(chunk)}\n\n"
                        except json.JSONDecodeError:
                            pass

                yield "data: [DONE]\n\n"

    except Exception as e:
        log.error(f"[{req_id}] Passthrough error: {e}")
        chunk = {
            "id": request_id, "object": "chat.completion.chunk",
            "created": created, "model": model_id,
            "choices": [{"index": 0, "delta": {"content": f"Error: {str(e)}"}, "finish_reason": "stop"}],
        }
        yield f"data: {json.dumps(chunk)}\n\n"
        yield "data: [DONE]\n\n"
    finally:
        active_requests.pop(req_id, None)


# ============================================================================
# FastAPI Endpoints
# ============================================================================

@app.get("/v1/models")
@app.get("/models")
async def list_models():
    return JSONResponse({
        "object": "list",
        "data": [{
            "id": "miroflow",
            "object": "model",
            "created": 1700000000,
            "owned_by": "deep-research-proxy",
            "name": "MiroFlow",
        }]
    })


@app.post("/v1/chat/completions")
@app.post("/chat/completions")
async def chat_completions(request: Request):
    req_id = f"req-{uuid.uuid4().hex[:8]}"

    try:
        body = await request.json()
    except Exception as e:
        return JSONResponse(
            status_code=400,
            content={"error": {"message": f"Invalid request body: {e}", "type": "invalid_request"}},
        )

    messages = body.get("messages", [])
    utility = is_utility_request(messages)

    log.info(f"[{req_id}] New request: messages={len(messages)}, model={body.get('model', '?')}, utility={utility}")

    active_requests[req_id] = {
        "started": datetime.now(timezone.utc).isoformat(),
        "utility": utility,
        "messages": len(messages),
        "current_turn": 0,
    }

    if utility:
        log.info(f"[{req_id}] Routing to PASSTHROUGH")
        generator = stream_passthrough(messages, body, req_id)
    else:
        log.info(f"[{req_id}] Routing to DEEP RESEARCH agent loop")
        generator = run_deep_research(messages, body, req_id)

    return StreamingResponse(
        generator,
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "service": "deep-research-proxy",
        "upstream": UPSTREAM_BASE,
        "upstream_model": UPSTREAM_MODEL,
        "searxng": SEARXNG_URL,
        "max_turns": MAX_AGENT_TURNS,
        "active_requests": len(active_requests),
        "active_details": active_requests,
    }


@app.get("/logs")
async def get_logs(lines: int = 100):
    log_path = os.path.join(LOG_DIR, "proxy.log")
    try:
        with open(log_path, "r") as f:
            all_lines = f.readlines()
            return JSONResponse({
                "total_lines": len(all_lines),
                "returned": min(lines, len(all_lines)),
                "lines": all_lines[-lines:],
            })
    except FileNotFoundError:
        return JSONResponse({"error": "Log file not found"}, status_code=404)


if __name__ == "__main__":
    import uvicorn
    log.info("Starting Deep Research Proxy (MiroFlow)...")
    uvicorn.run(app, host="0.0.0.0", port=LISTEN_PORT, log_level="info")
