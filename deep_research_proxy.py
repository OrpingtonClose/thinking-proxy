#!/usr/bin/env python3
"""
Deep Research Proxy for Open WebUI.

An OpenAI-compatible proxy that implements a MiroFlow-inspired agentic deep research
loop. When a user asks a question, the proxy orchestrates multi-turn reasoning with
tool use (SearXNG search, web page reading, Python execution) and streams the entire
research process as <think> tags to Open WebUI, followed by a polished final answer.

Architecture:
  - Receives OpenAI-compatible chat/completions requests from Open WebUI
  - Runs a multi-turn agent loop: LLM thinks → calls tools via XML → results fed back
  - All reasoning and tool interactions streamed as <think> content
  - Final answer streamed as main content after </think>
  - Utility requests (title/tag generation) bypass the agent loop

Tools (all self-hosted, no external APIs):
  - searxng_search: Search via local SearXNG instance
  - fetch_webpage: Fetch and extract readable text from URLs
  - python_exec: Execute Python code in a sandboxed subprocess
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
UPSTREAM_BASE = os.getenv("UPSTREAM_BASE", "https://openrouter.ai/api/v1")
UPSTREAM_KEY = os.getenv("UPSTREAM_KEY", "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")
UPSTREAM_MODEL = os.getenv("UPSTREAM_MODEL", "mistralai/mistral-large-2512")
SEARXNG_URL = os.getenv("SEARXNG_URL", "http://localhost:8888")
LISTEN_PORT = int(os.getenv("DEEP_RESEARCH_PORT", "9200"))
MAX_AGENT_TURNS = int(os.getenv("MAX_AGENT_TURNS", "15"))
WEBPAGE_MAX_CHARS = 15000
PYTHON_TIMEOUT = 30
PYTHON_OUTPUT_MAX = 5000

log.info(f"Config: model={UPSTREAM_MODEL}, searxng={SEARXNG_URL}, port={LISTEN_PORT}, max_turns={MAX_AGENT_TURNS}")

# --- Tool Definitions ---
TOOL_DEFINITIONS = """
## Server name: deep-research-tools

### Tool name: searxng_search
Description: Search the web using SearXNG. Returns top results with titles, URLs, and snippets. Use this to find information, verify facts, discover sources.
Input JSON schema: {"type": "object", "properties": {"query": {"type": "string", "description": "The search query"}}, "required": ["query"]}

### Tool name: fetch_webpage
Description: Fetch a webpage and extract its readable text content. Use this to read articles, documentation, or any web page found via search. Returns the main text content, truncated to ~15000 characters.
Input JSON schema: {"type": "object", "properties": {"url": {"type": "string", "description": "The URL to fetch"}, "extract_info": {"type": "string", "description": "Optional: specific information to look for in the page"}}, "required": ["url"]}

### Tool name: python_exec
Description: Execute Python code for calculations, data processing, or analysis. Code runs in a sandboxed subprocess with a 30-second timeout. Use for math, date calculations, data manipulation, etc.
Input JSON schema: {"type": "object", "properties": {"code": {"type": "string", "description": "Python code to execute. Use print() to output results."}}, "required": ["code"]}
"""

# --- System Prompt ---
SYSTEM_PROMPT_TEMPLATE = """In this environment you have access to a set of tools you can use to answer the user's question.

You only have access to the tools provided below. You can only use one tool per message, and will receive the result of that tool in the user's next response. You use tools step-by-step to accomplish a given task, with each tool-use informed by the result of the previous tool-use. Today is: {date}

# Tool-Use Formatting Instructions

Tool-use is formatted using XML-style tags. The tool-use is enclosed in <use_mcp_tool></use_mcp_tool> and each parameter is similarly enclosed within its own set of tags.

Parameters:
- server_name: (required) The name of the MCP server providing the tool
- tool_name: (required) The name of the tool to execute
- arguments: (required) A JSON object containing the tool's input parameters

Usage:
<use_mcp_tool>
<server_name>deep-research-tools</server_name>
<tool_name>tool name here</tool_name>
<arguments>
{{"param1": "value1"}}
</arguments>
</use_mcp_tool>

Important Notes:
- Tool-use must be placed **at the end** of your response, **top-level**, and not nested within other tags.
- Always adhere to this format for the tool use to ensure proper parsing and execution.
- You can only call ONE tool per message.

Here are the tools available:
{tools}

# Your Objective

You are a deep research agent. Your goal is to thoroughly research the user's question using the available tools and provide a comprehensive, well-sourced answer.

**Research Strategy:**
1. Break complex questions into sub-questions
2. Search for information using searxng_search
3. Read promising sources using fetch_webpage
4. Verify key claims by cross-referencing multiple sources
5. Use python_exec for any calculations or data processing needed
6. When you have gathered enough information, provide your final answer WITHOUT any tool call

**Important Rules:**
- Be thorough — search multiple angles, read multiple sources
- Be factual — cite your sources, don't speculate
- When your research is complete, write your final answer directly (no tool call)
- Your final answer should be comprehensive, well-structured, and directly address the user's question
- If you cannot find reliable information, say so honestly
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

app = FastAPI(title="Deep Research Proxy")
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
            # Remove script and style blocks
            text = re.sub(r'<script[^>]*>.*?</script>', '', raw_html, flags=re.DOTALL | re.IGNORECASE)
            text = re.sub(r'<style[^>]*>.*?</style>', '', raw_html, flags=re.DOTALL | re.IGNORECASE)
            # Remove HTML tags
            text = re.sub(r'<[^>]+>', ' ', text)
            # Decode HTML entities
            text = html.unescape(text)
            # Normalize whitespace
            text = re.sub(r'\s+', ' ', text).strip()
            # Collapse multiple newlines
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
# MCP XML Parsing
# ============================================================================

def parse_tool_call(text: str) -> Optional[dict]:
    """Parse a <use_mcp_tool> XML block from the LLM response."""
    match = re.search(
        r'<use_mcp_tool>\s*'
        r'<server_name>(.*?)</server_name>\s*'
        r'<tool_name>(.*?)</tool_name>\s*'
        r'<arguments>\s*(.*?)\s*</arguments>\s*'
        r'</use_mcp_tool>',
        text,
        re.DOTALL,
    )
    if not match:
        return None

    server_name = match.group(1).strip()
    tool_name = match.group(2).strip()
    arguments_str = match.group(3).strip()

    try:
        arguments = json.loads(arguments_str)
    except json.JSONDecodeError:
        # Try to fix common JSON issues
        try:
            # Sometimes the LLM uses single quotes
            arguments = json.loads(arguments_str.replace("'", '"'))
        except json.JSONDecodeError:
            return {"server_name": server_name, "tool_name": tool_name, "arguments": {}, "raw_args": arguments_str, "parse_error": True}

    return {"server_name": server_name, "tool_name": tool_name, "arguments": arguments}


def extract_text_before_tool_call(text: str) -> str:
    """Extract the reasoning text before any <use_mcp_tool> block."""
    match = re.search(r'<use_mcp_tool>', text)
    if match:
        return text[:match.start()].strip()
    return text.strip()


# ============================================================================
# LLM Communication
# ============================================================================

async def call_llm(messages: list[dict], req_id: str, turn: int) -> str:
    """Call the upstream LLM (non-streaming) and return the full response text."""
    body = {
        "model": UPSTREAM_MODEL,
        "messages": messages,
        "max_tokens": 4096,
        "temperature": 0.3,
        "stream": False,
    }

    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(300.0, connect=30.0)) as client:
            resp = await client.post(
                f"{UPSTREAM_BASE}/chat/completions",
                json=body,
                headers={
                    "Authorization": f"Bearer {UPSTREAM_KEY}",
                    "Content-Type": "application/json",
                    "HTTP-Referer": "https://deep-search.uk",
                    "X-Title": "Deep Research Proxy",
                },
            )

            if resp.status_code != 200:
                error_text = resp.text[:500]
                log.error(f"[{req_id}] Turn {turn}: LLM error {resp.status_code}: {error_text}")
                return f"[LLM Error: HTTP {resp.status_code}] {error_text}"

            data = resp.json()
            choices = data.get("choices", [])
            if not choices:
                return "[LLM Error: No choices in response]"

            content = choices[0].get("message", {}).get("content", "")
            # Strip <think> tags if the model uses them natively (Qwen3 does)
            content = re.sub(r'<think>.*?</think>\s*', '', content, flags=re.DOTALL)
            return content

    except httpx.ReadTimeout:
        return "[LLM Error: Request timed out]"
    except Exception as e:
        return f"[LLM Error: {str(e)}]"


async def call_llm_with_keepalive(
    messages: list[dict], req_id: str, turn: int, keepalive_queue: asyncio.Queue
) -> str:
    """Call LLM while sending keepalive signals so the SSE stream doesn't stall.
    Pushes a '.' to keepalive_queue every 8 seconds while waiting."""
    result_holder = {"value": None, "done": False}

    async def _do_call():
        result_holder["value"] = await call_llm(messages, req_id, turn)
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
    All agent reasoning goes inside <think>, final answer comes after </think>.
    """
    model_id = original_body.get("model", "deep-research")
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

    # Build system prompt with today's date and tools
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(date=today, tools=TOOL_DEFINITIONS)

    # Build message history for the agent
    agent_messages = [{"role": "system", "content": system_prompt}]
    # Add user messages (skip any system messages from Open WebUI)
    for msg in user_messages:
        if msg.get("role") != "system":
            agent_messages.append(msg)

    log.info(f"[{req_id}] Starting deep research loop, user messages: {len(user_messages)}")

    # Open the thinking block
    yield make_chunk("<think>\n")

    used_queries = set()  # Track duplicate searches
    used_urls = set()     # Track duplicate fetches
    keepalive_q = asyncio.Queue()

    async def llm_with_dots(msgs, turn_num):
        """Call LLM and collect keepalive dots to yield later."""
        return await call_llm_with_keepalive(msgs, req_id, turn_num, keepalive_q)

    def drain_keepalive():
        """Collect all pending keepalive dots into a single string."""
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

            # Call LLM with keepalive dots
            active_requests[req_id]["current_turn"] = turn
            llm_response = await llm_with_dots(agent_messages, turn)

            # Emit any accumulated keepalive dots
            dots = drain_keepalive()
            if dots:
                yield make_chunk(dots)

            # Check for LLM errors
            if llm_response.startswith("[LLM Error"):
                yield make_chunk(f"⚠️ {llm_response}\n")
                # Try to continue — the model might recover
                agent_messages.append({"role": "assistant", "content": llm_response})
                agent_messages.append({"role": "user", "content": "There was an error with your previous response. Please try a different approach."})
                continue

            # Parse for tool call
            tool_call = parse_tool_call(llm_response)
            reasoning = extract_text_before_tool_call(llm_response)

            if tool_call is None:
                # No tool call — this is the final answer
                log.info(f"[{req_id}] Turn {turn}: Final answer (no tool call)")
                # Stream the reasoning inside think block
                if reasoning:
                    yield make_chunk(f"Synthesizing final answer...\n")

                # Close the thinking block
                yield make_chunk("\n</think>\n\n")

                # Stream the final answer in chunks to avoid large single payload
                answer = llm_response
                chunk_size = 200
                for i in range(0, len(answer), chunk_size):
                    yield make_chunk(answer[i:i + chunk_size])

                yield make_chunk("", finish_reason="stop")
                yield "data: [DONE]\n\n"
                active_requests.pop(req_id, None)
                return

            # We have a tool call
            tool_name = tool_call["tool_name"]
            arguments = tool_call.get("arguments", {})

            # Stream reasoning
            if reasoning:
                # Truncate for the think stream (don't flood the UI)
                display_reasoning = reasoning[:500] + ("..." if len(reasoning) > 500 else "")
                yield make_chunk(f"{display_reasoning}\n\n")

            # Check for duplicates
            query_key = f"{tool_name}:{json.dumps(arguments, sort_keys=True)}"
            if query_key in used_queries:
                log.warning(f"[{req_id}] Turn {turn}: Duplicate tool call: {tool_name}")
                yield make_chunk(f"⚠️ Skipping duplicate {tool_name} call. Moving to answer...\n")
                # Force the model to give a final answer
                agent_messages.append({"role": "assistant", "content": llm_response})
                agent_messages.append({"role": "user", "content": "You have already made this exact tool call. Please provide your final answer now based on the information already gathered. Do NOT make any more tool calls."})
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

            # Check for parse errors
            if tool_call.get("parse_error"):
                yield make_chunk(f"⚠️ Could not parse tool arguments\n")
                agent_messages.append({"role": "assistant", "content": llm_response})
                agent_messages.append({"role": "user", "content": f"Your tool call had invalid JSON arguments. The raw arguments were: {tool_call.get('raw_args', '')}. Please fix the JSON and try again, or provide your answer."})
                continue

            # Execute tool
            tool_start = time.monotonic()
            tool_result = await execute_tool(tool_name, arguments)
            tool_duration = time.monotonic() - tool_start

            log.info(f"[{req_id}] Turn {turn}: Tool {tool_name} completed in {tool_duration:.1f}s, result length: {len(tool_result)}")

            # Stream brief result summary
            result_preview = tool_result[:300].replace('\n', ' ')
            yield make_chunk(f"→ Result ({tool_duration:.1f}s): {result_preview}{'...' if len(tool_result) > 300 else ''}\n")

            # Feed result back to agent
            agent_messages.append({"role": "assistant", "content": llm_response})
            agent_messages.append({"role": "user", "content": f"Tool result for {tool_name}:\n\n{tool_result}"})

        # Max turns reached — force answer
        log.info(f"[{req_id}] Max turns ({MAX_AGENT_TURNS}) reached, forcing final answer")
        yield make_chunk(f"\n⏰ Max research turns reached. Generating answer from gathered information...\n")

        # Ask model for final summary
        agent_messages.append({
            "role": "user",
            "content": "You have reached the maximum number of research turns. Based on ALL the information gathered so far, provide your final comprehensive answer to the original question. Do NOT call any tools."
        })
        final_response = await llm_with_dots(agent_messages, MAX_AGENT_TURNS + 1)
        dots = drain_keepalive()
        if dots:
            yield make_chunk(dots)

        yield make_chunk("\n</think>\n\n")
        # Stream final answer in chunks
        for i in range(0, len(final_response), 200):
            yield make_chunk(final_response[i:i + 200])
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
    """Pass utility requests straight through to OpenRouter without the agent loop."""
    model_id = original_body.get("model", "deep-research")
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
                    "HTTP-Referer": "https://deep-search.uk",
                    "X-Title": "Deep Research Proxy",
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
    log.info("Starting Deep Research Proxy...")
    uvicorn.run(app, host="0.0.0.0", port=LISTEN_PORT, log_level="info")
