#!/usr/bin/env python3
"""
Thinking Proxy for Mistral Large.

An OpenAI-compatible proxy that sits between Open WebUI and the Mistral API (via OpenRouter).
It injects a "think step-by-step" system instruction, then wraps the model's reasoning
output in <think>...</think> tags before streaming it back to the client.

The proxy implements a two-phase response:
  Phase 1 (Thinking): The model is asked to reason step-by-step in a structured way.
                       The proxy streams this wrapped in <think>...</think>.
  Phase 2 (Answer):   The model's final answer is streamed normally.

Architecture:
  - Receives OpenAI-compatible chat/completions requests
  - Detects utility requests (title/tag generation) and passes them through WITHOUT thinking
  - For real chat requests: injects thinking instructions, streams via state machine
  - All responses are ALWAYS streamed (SSE) regardless of client's stream param

Runs as a FastAPI app under uvicorn in a screen session.
"""

import asyncio
import json
import logging
import logging.handlers
import os
import re
import sys
import time
import traceback
import uuid
from datetime import datetime, timezone
from typing import AsyncGenerator, Optional

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, JSONResponse

# --- Logging ---
LOG_DIR = "/opt/thinking_proxy_logs"
os.makedirs(LOG_DIR, exist_ok=True)

# Setup logging
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

log = logging.getLogger("thinking-proxy")

# --- Configuration ---
UPSTREAM_BASE = os.getenv("UPSTREAM_BASE", "https://openrouter.ai/api/v1")
UPSTREAM_KEY = os.getenv("UPSTREAM_KEY", "")
UPSTREAM_MODEL = os.getenv("UPSTREAM_MODEL", "mistralai/mistral-large-2512")
LISTEN_PORT = int(os.getenv("THINKING_PROXY_PORT", "9100"))

log.info(f"Config: UPSTREAM_BASE={UPSTREAM_BASE}, UPSTREAM_MODEL={UPSTREAM_MODEL}, PORT={LISTEN_PORT}")

# The thinking instruction injected into the system prompt
THINKING_INSTRUCTION = """
IMPORTANT — STRUCTURED THINKING PROTOCOL:

You MUST structure every response in exactly two clearly labeled sections:

<THINKING>
[Your detailed step-by-step reasoning goes here. Break down the problem, consider different angles, 
evaluate evidence, check your logic, explore alternatives. Be thorough and genuine in your analysis.
This section should reflect your actual reasoning process — not a performance.]
</THINKING>

<ANSWER>
[Your final, polished response to the user goes here. This is what the user primarily cares about.
It should be comprehensive, well-structured, and directly address their question.]
</ANSWER>

You MUST always include both sections. The THINKING section comes first, then the ANSWER section.
Never skip the THINKING section. Never merge them. Always use the exact tags shown above.
"""

# Patterns to detect utility/automated requests from Open WebUI
# These should NOT get the thinking protocol — they need fast, clean responses
UTILITY_PATTERNS = [
    "generate a concise",
    "generate 1-3 broad tags",
    "generate a title",
    "### task:\ngenerate",
    "create a concise title",
    "generate a search query",
    "autocomplete",
]

app = FastAPI(title="Thinking Proxy")

# --- Request tracking ---
active_requests: dict[str, dict] = {}


def is_utility_request(messages: list[dict]) -> bool:
    """Detect if this is an automated utility request (title/tag generation, etc.)."""
    if not messages:
        return False
    # Check the last user message and system message for utility patterns
    for msg in messages:
        content = msg.get("content", "")
        if isinstance(content, str):
            content_lower = content.lower()
            for pattern in UTILITY_PATTERNS:
                if pattern in content_lower:
                    return True
    return False


def inject_thinking_prompt(messages: list[dict]) -> list[dict]:
    """Inject the thinking instruction into the message list."""
    messages = [m.copy() for m in messages]
    
    has_system = False
    for i, m in enumerate(messages):
        if m.get("role") == "system":
            messages[i]["content"] = m["content"] + "\n\n" + THINKING_INSTRUCTION
            has_system = True
            break
    
    if not has_system:
        messages.insert(0, {"role": "system", "content": THINKING_INSTRUCTION})
    
    return messages


async def stream_passthrough(
    messages: list[dict],
    original_body: dict,
    req_id: str,
) -> AsyncGenerator[str, None]:
    """
    Pass a request straight through to OpenRouter WITHOUT thinking injection.
    Used for utility requests (title generation, tag generation, etc.).
    Always streams the response as SSE.
    """
    model_id = original_body.get("model", UPSTREAM_MODEL)
    request_id = f"chatcmpl-pass-{uuid.uuid4().hex[:12]}"
    created = int(time.time())
    start_time = time.monotonic()

    # Build upstream request — NO thinking injection
    upstream_body = {
        **original_body,
        "model": UPSTREAM_MODEL,
        "messages": messages,  # Pass through as-is
        "stream": True,
    }
    # Remove keys that confuse OpenRouter or trigger tool_calls
    for key in ("user", "chat_id", "tools", "tool_choice", "functions", "function_call"):
        upstream_body.pop(key, None)

    log.info(f"[{req_id}] PASSTHROUGH upstream request: model={UPSTREAM_MODEL}, messages={len(messages)}")

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
                    "X-Title": "Deep Search Thinking Proxy",
                },
            ) as resp:
                if resp.status_code != 200:
                    error_body = await resp.aread()
                    error_text = error_body.decode("utf-8", errors="replace")[:1000]
                    log.error(f"[{req_id}] Passthrough upstream error {resp.status_code}: {error_text}")
                    chunk = {
                        "id": request_id, "object": "chat.completion.chunk",
                        "created": created, "model": model_id,
                        "choices": [{"index": 0, "delta": {"content": f"Error: {error_text[:200]}"}, "finish_reason": "stop"}],
                    }
                    yield f"data: {json.dumps(chunk)}\n\n"
                    yield "data: [DONE]\n\n"
                    return

                # Stream through directly — just relay SSE lines from OpenRouter
                async for line in resp.aiter_lines():
                    if line.startswith("data: "):
                        payload = line[6:].strip()
                        if payload == "[DONE]":
                            yield "data: [DONE]\n\n"
                            return
                        try:
                            chunk = json.loads(payload)
                            # Rewrite model name to match what Open WebUI expects
                            chunk["model"] = model_id
                            yield f"data: {json.dumps(chunk)}\n\n"
                        except json.JSONDecodeError:
                            pass

                # If we get here without [DONE], send it
                yield "data: [DONE]\n\n"

    except Exception as e:
        elapsed = time.monotonic() - start_time
        log.error(f"[{req_id}] Passthrough error after {elapsed:.2f}s: {e}")
        chunk = {
            "id": request_id, "object": "chat.completion.chunk",
            "created": created, "model": model_id,
            "choices": [{"index": 0, "delta": {"content": f"Error: {str(e)[:200]}"}, "finish_reason": "stop"}],
        }
        yield f"data: {json.dumps(chunk)}\n\n"
        yield "data: [DONE]\n\n"
    finally:
        active_requests.pop(req_id, None)


async def stream_thinking_response(
    messages: list[dict],
    original_body: dict,
    req_id: str,
) -> AsyncGenerator[str, None]:
    """
    Stream the upstream response, transforming <THINKING>/<ANSWER> tags
    into <think>/<content> format that Open WebUI renders natively.
    """
    model_id = original_body.get("model", UPSTREAM_MODEL)
    request_id = f"chatcmpl-think-{uuid.uuid4().hex[:12]}"
    created = int(time.time())
    token_count = 0
    start_time = time.monotonic()
    finish_sent = False  # Guard against duplicate finish chunks

    # Build upstream request — strip tools/functions to prevent tool_calls
    # The thinking proxy is text-only; tool calls would produce empty responses
    upstream_body = {
        **original_body,
        "model": UPSTREAM_MODEL,
        "messages": inject_thinking_prompt(messages),
        "stream": True,
    }
    # Remove keys that confuse OpenRouter or trigger tool_calls
    for key in ("user", "chat_id", "tools", "tool_choice", "functions", "function_call"):
        upstream_body.pop(key, None)

    tools_stripped = any(k in original_body for k in ("tools", "tool_choice", "functions", "function_call"))
    log.info(f"[{req_id}] THINKING upstream request: model={UPSTREAM_MODEL}, messages={len(messages)}, stream=True, tools_stripped={tools_stripped}")
    log.debug(f"[{req_id}] User message (last): {messages[-1].get('content', '')[:200]}")

    # State machine for parsing the stream
    buffer = ""
    phase = "pre_think"  # pre_think -> thinking -> between -> answering -> done
    think_opened = False
    think_closed = False

    def make_chunk(content: str, finish_reason: Optional[str] = None) -> str:
        """Create an SSE chunk in OpenAI streaming format."""
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

    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(300.0, connect=30.0)) as client:
            log.info(f"[{req_id}] Connecting to upstream: {UPSTREAM_BASE}/chat/completions")
            
            async with client.stream(
                "POST",
                f"{UPSTREAM_BASE}/chat/completions",
                json=upstream_body,
                headers={
                    "Authorization": f"Bearer {UPSTREAM_KEY}",
                    "Content-Type": "application/json",
                    "HTTP-Referer": "https://deep-search.uk",
                    "X-Title": "Deep Search Thinking Proxy",
                },
            ) as resp:
                elapsed_connect = time.monotonic() - start_time
                log.info(f"[{req_id}] Upstream responded: status={resp.status_code}, connect_time={elapsed_connect:.2f}s")
                
                if resp.status_code != 200:
                    error_body = await resp.aread()
                    error_text = error_body.decode("utf-8", errors="replace")[:1000]
                    log.error(f"[{req_id}] Upstream error {resp.status_code}: {error_text}")
                    
                    # Surface error clearly to the UI
                    error_msg = (
                        f"**Thinking Proxy Error**\n\n"
                        f"Upstream returned HTTP {resp.status_code}.\n\n"
                        f"```\n{error_text}\n```\n\n"
                        f"_Model: {UPSTREAM_MODEL} via {UPSTREAM_BASE}_"
                    )
                    yield make_chunk(error_msg)
                    yield make_chunk("", finish_reason="stop")
                    yield "data: [DONE]\n\n"
                    active_requests.pop(req_id, None)
                    return

                log.info(f"[{req_id}] Streaming started, phase={phase}")
                
                async for line in resp.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    
                    payload = line[6:].strip()
                    if payload == "[DONE]":
                        elapsed_total = time.monotonic() - start_time
                        log.info(f"[{req_id}] Stream complete: tokens={token_count}, total_time={elapsed_total:.2f}s, final_phase={phase}")
                        # Ensure we close thinking tag if still open
                        if think_opened and not think_closed:
                            yield make_chunk("\n</think>\n\n")
                        if not finish_sent:
                            yield make_chunk("", finish_reason="stop")
                            finish_sent = True
                        yield "data: [DONE]\n\n"
                        active_requests.pop(req_id, None)
                        return

                    try:
                        chunk = json.loads(payload)
                    except json.JSONDecodeError as e:
                        log.warning(f"[{req_id}] Bad JSON chunk: {e} — payload: {payload[:200]}")
                        continue

                    choices = chunk.get("choices", [])
                    if not choices:
                        continue
                    
                    delta = choices[0].get("delta", {})
                    token = delta.get("content", "")
                    
                    if not token:
                        fr = choices[0].get("finish_reason")
                        if fr:
                            log.info(f"[{req_id}] Finish reason received: {fr}")
                            if think_opened and not think_closed:
                                yield make_chunk("\n</think>\n\n")
                                think_closed = True
                            if not finish_sent:
                                yield make_chunk("", finish_reason="stop")
                                finish_sent = True
                        continue

                    token_count += 1
                    buffer += token

                    # --- State machine ---
                    
                    if phase == "pre_think":
                        if "<THINKING>" in buffer:
                            idx = buffer.index("<THINKING>") + len("<THINKING>")
                            remaining = buffer[idx:]
                            buffer = ""
                            yield make_chunk("<think>\n")
                            think_opened = True
                            log.info(f"[{req_id}] Phase: pre_think -> thinking (found <THINKING> at token {token_count})")
                            if remaining.strip():
                                if "</THINKING>" in remaining:
                                    think_part = remaining[:remaining.index("</THINKING>")]
                                    buffer = remaining[remaining.index("</THINKING>"):]
                                    if think_part.strip():
                                        yield make_chunk(think_part)
                                    phase = "between"
                                    log.info(f"[{req_id}] Phase: thinking -> between (immediate close)")
                                else:
                                    yield make_chunk(remaining)
                                    buffer = ""
                            phase = "thinking" if phase == "pre_think" else phase
                        else:
                            if len(buffer) > 200 and "<THINK" not in buffer:
                                log.warning(f"[{req_id}] Model not following protocol after 200 chars, forcing thinking phase")
                                yield make_chunk("<think>\n")
                                think_opened = True
                                yield make_chunk(buffer)
                                buffer = ""
                                phase = "thinking"

                    elif phase == "thinking":
                        if "</THINKING>" in buffer:
                            idx = buffer.index("</THINKING>")
                            think_content = buffer[:idx]
                            buffer = buffer[idx + len("</THINKING>"):]
                            if think_content:
                                yield make_chunk(think_content)
                            yield make_chunk("\n</think>\n\n")
                            think_closed = True
                            phase = "between"
                            log.info(f"[{req_id}] Phase: thinking -> between (at token {token_count})")
                        elif len(buffer) > 50:
                            emit = buffer[:-20]
                            buffer = buffer[-20:]
                            if emit:
                                yield make_chunk(emit)

                    elif phase == "between":
                        if "<ANSWER>" in buffer:
                            idx = buffer.index("<ANSWER>") + len("<ANSWER>")
                            remaining = buffer[idx:]
                            buffer = ""
                            log.info(f"[{req_id}] Phase: between -> answering (at token {token_count})")
                            if remaining.strip():
                                if "</ANSWER>" in remaining:
                                    answer_part = remaining[:remaining.index("</ANSWER>")]
                                    if answer_part.strip():
                                        yield make_chunk(answer_part)
                                    phase = "done"
                                    log.info(f"[{req_id}] Phase: answering -> done (immediate close)")
                                else:
                                    yield make_chunk(remaining)
                                    buffer = ""
                                    phase = "answering"
                            else:
                                phase = "answering"
                        elif len(buffer) > 200:
                            log.warning(f"[{req_id}] No <ANSWER> tag found after 200 chars, streaming directly")
                            emit = buffer[:-20]
                            buffer = buffer[-20:]
                            if emit.strip():
                                yield make_chunk(emit)
                            phase = "answering"

                    elif phase == "answering":
                        if "</ANSWER>" in buffer:
                            idx = buffer.index("</ANSWER>")
                            answer_content = buffer[:idx]
                            buffer = ""
                            if answer_content:
                                yield make_chunk(answer_content)
                            phase = "done"
                            log.info(f"[{req_id}] Phase: answering -> done (at token {token_count})")
                        elif len(buffer) > 50:
                            emit = buffer[:-20]
                            buffer = buffer[-20:]
                            if emit:
                                yield make_chunk(emit)

                    elif phase == "done":
                        buffer = ""

    except httpx.ConnectError as e:
        elapsed = time.monotonic() - start_time
        log.error(f"[{req_id}] Connection error after {elapsed:.2f}s: {e}")
        error_msg = (
            f"**Thinking Proxy — Connection Error**\n\n"
            f"Could not connect to upstream: `{UPSTREAM_BASE}`\n\n"
            f"```\n{str(e)}\n```"
        )
        yield make_chunk(error_msg)
        if not finish_sent:
            yield make_chunk("", finish_reason="stop")
        yield "data: [DONE]\n\n"

    except httpx.ReadTimeout as e:
        elapsed = time.monotonic() - start_time
        log.error(f"[{req_id}] Read timeout after {elapsed:.2f}s (tokens so far: {token_count}): {e}")
        timeout_msg = (
            f"\n\n**Thinking Proxy — Timeout**\n\n"
            f"Upstream stopped responding after {elapsed:.1f}s ({token_count} tokens received).\n"
            f"The model may be overloaded. Try again."
        )
        if think_opened and not think_closed:
            yield make_chunk("\n</think>\n\n")
        yield make_chunk(timeout_msg)
        if not finish_sent:
            yield make_chunk("", finish_reason="stop")
        yield "data: [DONE]\n\n"

    except httpx.TimeoutException as e:
        elapsed = time.monotonic() - start_time
        log.error(f"[{req_id}] Timeout after {elapsed:.2f}s: {e}")
        error_msg = (
            f"**Thinking Proxy — Timeout**\n\n"
            f"Request timed out after {elapsed:.1f}s.\n\n"
            f"```\n{str(e)}\n```"
        )
        if think_opened and not think_closed:
            yield make_chunk("\n</think>\n\n")
        yield make_chunk(error_msg)
        if not finish_sent:
            yield make_chunk("", finish_reason="stop")
        yield "data: [DONE]\n\n"

    except Exception as e:
        elapsed = time.monotonic() - start_time
        tb = traceback.format_exc()
        log.error(f"[{req_id}] Unhandled exception after {elapsed:.2f}s: {e}\n{tb}")
        error_msg = (
            f"**Thinking Proxy — Internal Error**\n\n"
            f"An unexpected error occurred:\n\n"
            f"```\n{type(e).__name__}: {str(e)}\n```\n\n"
            f"_Check proxy logs for details (request: {req_id})_"
        )
        if think_opened and not think_closed:
            yield make_chunk("\n</think>\n\n")
        yield make_chunk(error_msg)
        if not finish_sent:
            yield make_chunk("", finish_reason="stop")
        yield "data: [DONE]\n\n"

    finally:
        active_requests.pop(req_id, None)

    # Flush any remaining buffer
    if buffer.strip():
        if think_opened and not think_closed:
            yield make_chunk(buffer)
            yield make_chunk("\n</think>\n\n")
        else:
            yield make_chunk(buffer)


@app.get("/v1/models")
@app.get("/models")
async def list_models():
    """Return the thinking model in OpenAI format."""
    return JSONResponse({
        "object": "list",
        "data": [
            {
                "id": "mistral-large-thinking",
                "object": "model",
                "created": 1700000000,
                "owned_by": "thinking-proxy",
                "name": "Mistral Large (Thinking)",
            }
        ]
    })


@app.post("/v1/chat/completions")
@app.post("/chat/completions")
async def chat_completions(request: Request):
    """Handle chat completion requests with thinking injection."""
    req_id = f"req-{uuid.uuid4().hex[:8]}"
    
    try:
        body = await request.json()
    except Exception as e:
        log.error(f"[{req_id}] Failed to parse request body: {e}")
        return JSONResponse(
            status_code=400,
            content={"error": {"message": f"Invalid request body: {e}", "type": "invalid_request"}},
        )
    
    messages = body.get("messages", [])
    client_wants_stream = body.get("stream", False)
    utility = is_utility_request(messages)
    
    log.info(f"[{req_id}] New request: client_stream={client_wants_stream}, messages={len(messages)}, model={body.get('model', '?')}, utility={utility}")
    
    # ALWAYS stream responses — the thinking proxy needs streaming for the state machine,
    # and non-streaming causes the UI to hang while waiting for the full response.
    if not client_wants_stream:
        log.info(f"[{req_id}] Overriding stream=false -> stream=true (proxy always streams)")
    
    # Track active request
    active_requests[req_id] = {
        "started": datetime.now(timezone.utc).isoformat(),
        "stream": True,
        "utility": utility,
        "messages": len(messages),
    }

    # Choose handler based on request type
    if utility:
        log.info(f"[{req_id}] Routing to PASSTHROUGH (utility request — no thinking injection)")
        generator = stream_passthrough(messages, body, req_id)
    else:
        log.info(f"[{req_id}] Routing to THINKING pipeline")
        generator = stream_thinking_response(messages, body, req_id)

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
        "service": "thinking-proxy",
        "upstream": UPSTREAM_BASE,
        "upstream_model": UPSTREAM_MODEL,
        "active_requests": len(active_requests),
        "active_details": active_requests,
    }


@app.get("/logs")
async def get_logs(lines: int = 100):
    """Return the last N lines of the proxy log file."""
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
    log.info("Starting Thinking Proxy...")
    uvicorn.run(app, host="0.0.0.0", port=LISTEN_PORT, log_level="info")
