# Scalable Agentic System (retrieval-based tool routing)

Setup: `pip install openai` and set `GROQ_API_KEY` (required).
Run: `python demo.py` or `python main.py` for chat.

## Core idea
The LLM never sees all tools. Per turn:

```
user msg -> Tool Retriever (top-K of N) -> LLM sees K tools only -> validate args -> execute -> loop/answer
                                                  + rag_search + system_search are ordinary tools in the same index
```

Accuracy depends on K (about 5), not N. Going from 50 to 5000 tools only grows the index.

## Components (`agent/`)
- `tools.py`: Tool = name, description, param schema, handler. `load_postman()` builds tools from a Postman collection. Includes `rag_search` (knowledge base) and `system_search` (capabilities and request log).
- `retriever.py`: TF-IDF retriever over tool name, description, and tags.
- `agent.py`: `ToolRegistry`, the Groq LLM backend, and the agent loop.

## Design answers
- **Agent structure**: one agent with a tool loop. A single agent is enough because retrieval keeps the choice small. For 500+ APIs across services, add a router that picks a service (PayPal, Stripe...) and then a per-service agent, which is still the same retrieval idea.
- **Tool selection and routing**: hybrid retrieval (BM25/TF-IDF plus embeddings, then optional reranker) over tool descriptions. Pin a few always-available tools (RAG, system search). Index quality matters, so write good descriptions and example queries per tool.
- **State management**: the loop keeps `history` of tool results per request. Production: persist conversation and results by session in Redis/Postgres (LangGraph checkpointer). `REQUEST_LOG` backs "status of my last request".
- **Scalability**: tools are data, not prompt text. Use a vector DB (pgvector/Qdrant) for thousands of tools. The agent is stateless, so scale it horizontally. Cache embeddings and tool schemas. Cap steps and tokens.
- **Error handling**: missing required params make the agent ask the user instead of guessing. A hallucinated tool name (not in the shortlist) is excluded and retried. Handler exceptions are retried once, then the tool is excluded so the agent tries another. A step cap prevents loops. Write actions (payments, refunds) should require user confirmation before execution (not implemented in this demo).

## Framework choice and trade-offs
| Option | Pros | Cons |
|---|---|---|
| **From scratch (used here)** | Tiny, transparent, no deps, easy to test | You build persistence, streaming, and retries yourself |
| LangGraph | Checkpointed state, human-in-the-loop, graph control flow | More concepts. Overkill for a simple loop |
| LangChain / LlamaIndex | Many integrations, strong RAG tools (LlamaIndex) | Heavy abstractions, harder to debug |
| CrewAI | Multi-agent roles | Extra LLM calls and latency. Not needed for tool routing |
| DSPy | Optimizes prompts against a metric | Needs a labelled eval set |

I'd move to LangGraph in production for checkpointing and confirmation steps, keeping this retriever as the routing node.

## Observability
Use LangSmith (set `LANGSMITH_TRACING=true`, wrap `Agent.run` and `_execute` with `@traceable`) or Langfuse/OpenTelemetry. Track retrieval recall@K (was the right tool in the shortlist?), tool-call success rate, latency, and cost.

## Real-LLM behavior
- `Agent` keeps the conversation per instance (last 20 messages). If it asks for a missing value, the next message is combined with the original request, so "bob@x.com" completes "Send an invoice for $75".
- `GroqLLM` uses SDK retries with backoff (`max_retries=3`, 30s timeout). Any remaining LLM failure returns a friendly message instead of crashing.
- Tool results are sent in the standard format (assistant `tool_calls`, then a `tool` message with the matching id).

## Notes for developers

### LLM backend contract
Any backend passed as `Agent(llm=...)` implements:

```python
def decide(self, query, conversation, tools, history) -> dict:
    # returns {"tool": name, "args": {...}} or {"final": text}
```

- `query`: text used for tool retrieval and parameter extraction. It includes the earlier request when the agent is waiting on a missing value.
- `conversation`: `[{"role": "user" | "assistant", "content": str}]`, the last 20 messages.
- `tools`: the shortlist (about 5 tools). Send only these to the model, never the full catalog.
- `history`: `[{"id", "tool", "args", "result"}]`, tool calls made so far for the current request.

### Enabling the real LLM
1. Python 3.9 or newer, and `pip install openai` (pin the version).
2. Set `GROQ_API_KEY` (never commit it). The `openai` SDK is pointed at Groq's OpenAI-compatible endpoint.
3. The model is `openai/gpt-oss-20b` by default (`GroqLLM.__init__` in `agent/agent.py`). It must support tool calling. To use OpenAI itself, change `base_url`, the key variable and the model name.
4. Run `python demo.py` and check each message calls the expected tool (create_invoice, sales_report, list_disputes, system_search, rag_search). A request with a missing email should ask for it.

### Using your own Postman collection
`load_postman()` in `agent/tools.py` builds tools with names and descriptions only, no parameters. Replace `build_api_tools()` in `Agent.__init__` with it, and extend it to parse each request's URL and body into parameters and real HTTP handlers (with PayPal OAuth credentials).

### Open items
- Only the first tool call per step is used if the model returns several.
- No user confirmation before payments and refunds.
- Retrieval is word-based. Use embeddings so synonyms ("bill someone") find the right tool.
- Conversation and request log are in memory only. Persist them per session for production.
- The Groq path has not been run against the real API by the author of this code.

## Limitations
The TF-IDF retriever is a simple stand-in. The PayPal handlers are mocks, not real API calls. Single tool call per step.
