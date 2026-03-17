# Thinking Proxy & Deep Research Proxy for Open WebUI

Two FastAPI proxy services that extend [Open WebUI](https://github.com/open-webui/open-webui) with structured reasoning capabilities. Both sit between Open WebUI and upstream LLM providers, transforming responses into Open WebUI's native `<think>` block format.

---

## Thinking Proxy (`thinking_proxy.py`)

Adds visible chain-of-thought reasoning to any OpenAI-compatible model by injecting a structured thinking protocol.

### How it works

1. **Receives** an OpenAI-compatible `/v1/chat/completions` request from Open WebUI
2. **Detects** if the request is a utility call (title/tag generation) — if so, passes through without modification
3. **Injects** a structured thinking protocol into the system prompt, instructing the model to respond with `<THINKING>` and `<ANSWER>` sections
4. **Streams** the response from the upstream provider
5. **Transforms** `<THINKING>/<ANSWER>` tags into `<think>/<content>` format via a state machine parser
6. **Always returns SSE** (`text/event-stream`) regardless of the client's `stream` parameter

### Key features

- **Streaming state machine**: Parses `<THINKING>`/`</THINKING>`/`<ANSWER>`/`</ANSWER>` tags as they arrive token-by-token
- **Utility request bypass**: Auto-detects title/tag generation and other automated Open WebUI requests
- **Tool stripping**: Removes `tools`/`functions` from requests to prevent `finish_reason: tool_calls`
- **Force-streaming**: Always returns SSE even when Open WebUI sends `stream=false`

### Environment variables

| Variable | Default | Description |
|---|---|---|
| `UPSTREAM_BASE` | `https://openrouter.ai/api/v1` | Upstream OpenAI-compatible API |
| `UPSTREAM_KEY` | *(none)* | API key for upstream |
| `UPSTREAM_MODEL` | `mistralai/mistral-large-2512` | Model to forward to |
| `THINKING_PROXY_PORT` | `9100` | Listen port |

---

## Deep Research Proxy (`deep_research_proxy.py`) — MiroFlow

A MiroFlow-inspired agentic deep research engine. Runs a multi-turn research loop with self-hosted tools, streaming all reasoning as `<think>` tags.

### How it works

1. **Receives** an OpenAI-compatible request from Open WebUI
2. **Detects** utility requests (title/tag generation) — bypasses agent loop
3. **Runs an agent loop** (up to 15 turns):
   - LLM reasons about the question
   - Calls tools via MCP-style XML (`<use_mcp_tool>`)
   - Tool results fed back to continue research
   - All reasoning streamed inside `<think>` tags
4. **Final answer** streamed after `</think>` when the agent is done

### Tools (all self-hosted)

| Tool | Description |
|---|---|
| `searxng_search` | Search via local SearXNG instance (localhost:8888) |
| `fetch_webpage` | Fetch and extract readable text from URLs |
| `python_exec` | Execute Python code in a sandboxed subprocess |

### Key features

- **MCP-style XML tool calling**: Uses `<use_mcp_tool>` format for reliable tool invocation
- **Duplicate detection**: Prevents infinite loops from repeated searches/fetches
- **Comprehensive logging**: Rotating logs at `/opt/deep_research_logs/proxy.log`
- **Error resilience**: LLM errors trigger retry with different approach
- **Utility bypass**: Automated Open WebUI requests skip the agent loop
- **Anti-censorship**: All tools self-hosted, no external API dependencies for search

### Environment variables

| Variable | Default | Description |
|---|---|---|
| `UPSTREAM_BASE` | `https://openrouter.ai/api/v1` | Upstream LLM API |
| `UPSTREAM_KEY` | *(none)* | API key for upstream |
| `UPSTREAM_MODEL` | `mistralai/mistral-large-2512` | Agent model |
| `SEARXNG_URL` | `http://localhost:8888` | SearXNG instance |
| `DEEP_RESEARCH_PORT` | `9200` | Listen port |
| `MAX_AGENT_TURNS` | `15` | Max research turns per request |

---

## Deployment

### Requirements

```
pip install fastapi uvicorn httpx
```

### Run both proxies

```bash
# Thinking Proxy
screen -dmS thinking-proxy python3 thinking_proxy.py

# Deep Research Proxy (MiroFlow)
screen -dmS deep-research python3 deep_research_proxy.py
```

### Open WebUI configuration

Add both as OpenAI-compatible providers:

| Index | Service | URL |
|---|---|---|
| 5 | Thinking Proxy | `http://localhost:9100/v1` |
| 6 | Deep Research Proxy | `http://localhost:9200/v1` |

Create model aliases:

| Alias | Base Model | Name |
|---|---|---|
| `mistral-large-thinking-proxy` | `mistral-large-thinking` | Mistral Large (Thinking) |
| `mistral-large-miroflow` | `miroflow` | Mistral Large (MiroFlow) |

## Architecture

```
                          ┌─ Thinking Proxy (:9100) ──→ OpenRouter ──→ Mistral Large
                          │   Inject thinking protocol, parse <THINKING>/<ANSWER>
                          │
User → Open WebUI (:3000) ┤
                          │
                          └─ Deep Research Proxy (:9200) ──→ OpenRouter ──→ Mistral Large
                              Multi-turn agent loop:          ↕
                              ├── SearXNG search (:8888)     Tool results
                              ├── Web page fetching          fed back
                              └── Python execution           per turn
```

## License

MIT
