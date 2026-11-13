import json, logging, os, time
from openai import OpenAI
from .retriever import Retriever
from .tools import Tool, REQUEST_LOG, build_api_tools, rag_search, make_system_search

log = logging.getLogger("agent")

TOP_K = 5
MAX_STEPS = 4
MAX_CONVERSATION = 20


# Indexes all tools and returns a small shortlist per query
class ToolRegistry:
    def __init__(self, tools: list[Tool]):
        self.tools = {t.name: t for t in tools}
        self.names = list(self.tools)
        self.index = Retriever([self.tools[n].text() for n in self.names])

    def search(self, query: str, k: int = TOP_K, exclude=()) -> list[Tool]:
        hits = [self.tools[self.names[i]] for i, _ in self.index.search(query, k + len(exclude))]
        return [t for t in hits if t.name not in exclude][:k]


# Groq LLM (OpenAI-compatible function calling); only the shortlisted tools are sent
class GroqLLM:
    SYSTEM = ("You are an assistant that uses tools to act on a PayPal account. "
              "Use a tool when needed. Never invent parameter values; if a required value is missing, "
              "ask the user for it. After tool results arrive, answer the user concisely.")

    def __init__(self, model="openai/gpt-oss-20b"):
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError("GROQ_API_KEY is not set")
        self.client = OpenAI(
            base_url="https://api.groq.com/openai/v1",
            api_key=api_key,
            max_retries=3, timeout=30,
        )
        self.model = model

    def decide(self, query, conversation, tools, history):
        schema = [{"type": "function", "function": {
            "name": t.name, "description": t.description,
            "parameters": {"type": "object",
                           "properties": {k: {"type": v["type"]} for k, v in t.params.items()},
                           "required": [k for k, v in t.params.items() if v["required"]]}}} for t in tools]
        msgs = [{"role": "system", "content": self.SYSTEM}] + list(conversation)
        for h in history:
            msgs.append({"role": "assistant", "content": None, "tool_calls": [{
                "id": h["id"], "type": "function",
                "function": {"name": h["tool"], "arguments": json.dumps(h["args"])}}]})
            msgs.append({"role": "tool", "tool_call_id": h["id"], "content": json.dumps(h["result"])})
        kwargs = {"tools": schema} if schema else {}
        r = self.client.chat.completions.create(model=self.model, messages=msgs, **kwargs).choices[0].message
        if r.tool_calls:
            c = r.tool_calls[0].function
            return {"tool": c.name, "args": json.loads(c.arguments or "{}")}
        return {"final": r.content or ""}


# One Agent instance = one chat session
class Agent:
    def __init__(self, llm=None):
        query_param = {"query": {"type": "string", "required": True}}
        tools = build_api_tools() + [
            Tool("rag_search", "Search knowledge base docs guides policy rules how to", query_param,
                 rag_search, ["documentation", "policy", "can i", "within"]),
            Tool("system_search", "Search available tools capabilities and request status history", query_param,
                 make_system_search(lambda q, k: self.registry.search(q, k)),
                 ["what tools available", "status", "last request"]),
        ]
        self.registry = ToolRegistry(tools)
        self.llm = llm or GroqLLM()
        self.conversation: list[dict] = []
        self.pending = None

    def _reply(self, text: str) -> str:
        self.conversation.append({"role": "assistant", "content": text})
        del self.conversation[:-MAX_CONVERSATION]
        return text

    def run(self, message: str) -> str:
        self.conversation.append({"role": "user", "content": message})
        query = f"{self.pending} {message}" if self.pending else message
        self.pending = None
        history, excluded = [], set()
        for step in range(MAX_STEPS):
            # retrieve shortlist, then let the LLM decide
            shortlist = self.registry.search(query, TOP_K, excluded)
            try:
                d = self.llm.decide(query, self.conversation, shortlist, history)
            except Exception:
                log.exception("LLM call failed")
                return self._reply("Sorry, I couldn't reach the language model. Please try again in a moment.")
            if "final" in d:
                return self._reply(d["final"])
            # validate the tool and its params; ask the user instead of guessing
            tool = self.registry.tools.get(d["tool"])
            if tool is None or tool not in shortlist:
                excluded.add(d["tool"])
                continue
            args = d.get("args") or {}
            missing = [p for p, s in tool.params.items() if s["required"] and args.get(p) in (None, "")]
            if missing:
                self.pending = query
                return self._reply(f"To use {tool.name} I need: {', '.join(missing)}.")
            # execute; on error, exclude the tool so another is tried
            start = time.perf_counter()
            result = self._execute(tool, args)
            log.info("tool=%s ok=%s ms=%d shortlist=%s", tool.name, "error" not in result,
                     (time.perf_counter() - start) * 1000, [t.name for t in shortlist])
            REQUEST_LOG.append({"message": message, "tool": tool.name, "ok": "error" not in result})
            if "error" in result:
                excluded.add(tool.name)
            history.append({"id": f"call_{step}", "tool": tool.name, "args": args, "result": result})
        return self._reply("Sorry, I couldn't complete that request.")

    @staticmethod
    def _execute(tool: Tool, args: dict, retries: int = 1) -> dict:
        for _ in range(retries + 1):
            try:
                return tool.handler(**args)
            except Exception as e:
                err = str(e)
        return {"error": err}
