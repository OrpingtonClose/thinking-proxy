# Thinking Proxy for Open WebUI

A FastAPI proxy that adds visible chain-of-thought reasoning to any OpenAI-compatible model. Sits between [Open WebUI](https://github.com/open-webui/open-webui) and an upstream provider (e.g. OpenRouter), injecting a structured thinking protocol and transforming the response into Open WebUI's native `<think>` block format.

## How it works

1. **Receives** an OpenAI-compatible `/v1/chat/completions` request from Open WebUI
2. **Detects** if the request is a utility call (title/tag generation) — if so, passes through without modification
3. **Injects** a structured thinking protocol into the system prompt, instructing the model to respond with `<THINKING>` and `<ANSWER>` sections
4. **Streams** the response from the upstream provider
5. **Transforms** `<THINKING>/<ANSWER>` tags into `<think>/<content>` format via a state machine parser
6. **Always returns SSE** (`text/event-stream`) regardless of the client's `stream` parameter — required for Open WebUI's thinking block rendering

## Key features

- **Streaming state machine**: Parses `<THINKING>`/`</THINKING>`/`<ANSWER>`/`</ANSWER>` tags as they arrive token-by-token, with buffering and fallback handling
- **Utility request bypass**: Auto-detects title generation, tag generation, and other automated Open WebUI requests — passes them through without the thinking protocol
- **Tool stripping**: Removes `tools`/`functions` from requests to prevent `finish_reason: tool_calls` with empty content
- **Force-streaming**: Always returns SSE even when Open WebUI sends `stream=false` — Open WebUI detects streaming from the response `Content-Type` header
- **Comprehensive logging**: Rotating log files at `/opt/thinking_proxy_logs/proxy.log`, per-request tracking with phase transitions
- **Error surfacing**: Upstream errors, timeouts, and connection failures are rendered as formatted markdown in the chat UI
- **Health & debug endpoints**: `/health` shows active requests, `/logs` returns recent log lines

## Deployment

### Requirements

```
pip install fastapi uvicorn httpx
```

### Environment variables

| Variable | Default | Description |
|---|---|---|
| `UPSTREAM_BASE` | `https://openrouter.ai/api/v1` | Base URL of the upstream OpenAI-compatible API |
| `UPSTREAM_KEY` | *(none)* | API key for the upstream provider |
| `UPSTREAM_MODEL` | `mistralai/mistral-large-2512` | Model ID to forward requests to |
| `THINKING_PROXY_PORT` | `9100` | Port to listen on |

### Run

```bash
python3 thinking_proxy.py
```

Or in a screen session:

```bash
screen -dmS thinking-proxy python3 /opt/thinking_proxy.py
```

### Open WebUI configuration

1. Add the proxy as an OpenAI-compatible provider:
   - URL: `http://localhost:9100/v1`
   - API key: any non-empty string

2. In `OPENAI_API_CONFIGS`, register the model:
   ```json
   {"5": {"enable": true, "model_ids": ["mistral-large-thinking"]}}
   ```

3. Create a model alias in Open WebUI:
   - **Alias ID**: `mistral-large-thinking-proxy` (must differ from base model ID)
   - **Base Model ID**: `mistral-large-thinking`
   - **Name**: Mistral Large (Thinking)

## Architecture

```
User → Open WebUI → Thinking Proxy (:9100) → OpenRouter → Mistral Large
                         │
                         ├── Utility request? → Passthrough (no thinking injection)
                         │
                         └── Chat request? → Inject thinking protocol
                                               → Stream & parse with state machine
                                               → Transform <THINKING>→<think>, <ANSWER>→content
                                               → SSE back to Open WebUI
```

## License

MIT
