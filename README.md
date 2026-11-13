# Scalable Agentic System (retrieval-based tool routing)

Run: `python demo.py` (offline mock LLM) or `python main.py` for chat.
With `GROQ_API_KEY` set and `pip install openai`, a real LLM (Groq) is used automatically.


## Real-LLM behavior
- `Agent` keeps the conversation per instance (last 20 messages). If it asks for a missing value, the next message is combined with the original request, so "bob@x.com" completes "Send an invoice for $75".
- `OpenAILLM` uses SDK retries with backoff (`max_retries=3`, 30s timeout). Any remaining LLM failure returns a friendly message instead of crashing.
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
2. Set `OPENAI_API_KEY` (never commit it). `OPENAI_BASE_URL` is optional, for a proxy or compatible gateway.
3. The model is `gpt-4o-mini` by default (`OpenAILLM.__init__` in `agent/agent.py`). It must support tool calling. For Azure OpenAI, switch the client to `AzureOpenAI` and use the deployment name.
4. Run `python demo.py` and check each message calls the expected tool (create_invoice, sales_report, list_disputes, system_search, rag_search). A request with a missing email should ask for it.

### Using your own Postman collection
`load_postman()` in `agent/tools.py` builds tools with names and descriptions only, no parameters. Replace `build_api_tools()` in `Agent.__init__` with it, and extend it to parse each request's URL and body into parameters and real HTTP handlers (with PayPal OAuth credentials).

### Open items
- Only the first tool call per step is used if the model returns several.
- No user confirmation before payments and refunds.
- Retrieval is word-based. Use embeddings so synonyms ("bill someone") find the right tool.
- Conversation and request log are in memory only. Persist them per session for production.
- The OpenAI path has not been run against the real API.

## Limitations
The mock LLM and TF-IDF retriever are simple stand-ins. The PayPal handlers are mocks, not real API calls. Single tool call per step. Multi-step sequences work through the loop only with a real LLM.
2. Set `GROQ_API_KEY` (never commit it). The `openai` SDK is pointed at Groq's OpenAI-compatible endpoint.
3. The model is `openai/gpt-oss-20b` by default (`OpenAILLM.__init__` in `agent/agent.py`). It must support tool calling. To use OpenAI itself, change `base_url`, the key variable and the model name.