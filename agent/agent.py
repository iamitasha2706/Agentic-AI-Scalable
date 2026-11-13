import json, logging, os, re
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


# Offline LLM stand-in: picks the top tool and extracts params with regex
class MockLLM:
    def decide(self, query, conversation, tools, history):
        if history:
            return {"final": f"Done. Result: {json.dumps(history[-1]['result'])}"}
        if not tools:
            return {"final": "I couldn't find a tool for that."}
        tool = tools[0]
        args = {}
        email = re.search(r"[\w.+-]+@[\w-]+\.[\w.]+", query)
        amount = re.search(r"\$?(\d+(?:\.\d+)?)", query)
        uid = re.search(r"user_\w+", query)
        for p in tool.params:
            if p == "query": args[p] = query
            elif p == "email" and email: args[p] = email.group()
            elif p == "amount" and amount: args[p] = float(amount.group(1))
            elif p == "user_id" and uid: args[p] = uid.group()
            elif p == "period": args[p] = "last_month" if "last month" in query.lower() else "all"
        return {"tool": tool.name, "args": args}


# Real LLM via OpenAI function calling; only the shortlisted tools are sent
class OpenAILLM:
    SYSTEM = ("You are an assistant that uses tools to act on a PayPal account. "
              "Use a tool when needed. Never invent parameter values; if a required value is missing, "
              "ask the user for it. After tool results arrive, answer the user concisely.")

    def __init__(self, model="openai/gpt-oss-20b"):
        from openai import OpenAI
        self.client = OpenAI(
            base_url="https://api.groq.com/openai/v1",
            api_key=os.getenv("GROQ_API_KEY"),
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
        tools = build_api_tools()
        reg_holder = {}
        tools += [
            Tool("rag_search", "Search knowledge base docs guides policy rules how to", {"query": {"type": "string", "required": True}},
                 lambda query: rag_search(query), ["documentation", "policy", "can i", "within"]),
            Tool("system_search", "Search available tools capabilities and request status history",
                 {"query": {"type": "string", "required": True}},
                 lambda query: make_system_search(lambda q, k: reg_holder["r"].search(q, k))(query),
                 ["what tools available", "status", "last request"]),
        ]
        self.registry = reg_holder["r"] = ToolRegistry(tools)
        self.llm = llm or (OpenAILLM() if os.getenv("GROQ_API_KEY") else MockLLM())
        self.conversation: list[dict] = []
        self.pending: str | None = None

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
            result = self._execute(tool, args)
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
