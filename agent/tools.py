from dataclasses import dataclass, field
from typing import Callable
import json

from .retriever import Retriever

REQUEST_LOG: list[dict] = []


@dataclass
class Tool:
    name: str
    description: str
    params: dict
    handler: Callable[..., dict]
    tags: list[str] = field(default_factory=list)

    def text(self) -> str:
        return f"{self.name.replace('_', ' ')} {self.description} {' '.join(self.tags)} {' '.join(self.params)}"


def _mock(name):
    return lambda **kw: {"status": "ok", "api": name, "input": kw}


# Mock PayPal catalog: (name, description, params, tags)
_CATALOG = [
    ("create_invoice", "Create and send an invoice to a customer email for an amount", {"email": "string", "amount": "number"}, ["bill", "send invoice"]),
    ("get_invoice", "Get details of an invoice by id", {"invoice_id": "string"}, []),
    ("cancel_invoice", "Cancel an invoice by id", {"invoice_id": "string"}, []),
    ("list_invoices", "List all invoices", {}, ["invoice"]),
    ("send_payment", "Send a payout payment to a recipient email", {"email": "string", "amount": "number"}, ["payout", "pay"]),
    ("refund_payment", "Refund a captured payment", {"capture_id": "string"}, []),
    ("list_disputes", "List open disputes, optionally filtered by user id", {"user_id": "string"}, ["dispute", "chargeback"]),
    ("get_dispute", "Get dispute details by dispute id", {"dispute_id": "string"}, []),
    ("accept_dispute_claim", "Accept the claim of a dispute", {"dispute_id": "string"}, []),
    ("sales_report", "Get total sales volume and transaction report for a period", {"period": "string"}, ["revenue", "transactions", "total sales"]),
    ("get_balance", "Get account balance", {}, ["money", "funds"]),
    ("create_subscription", "Create a subscription plan for a customer", {"email": "string", "plan_id": "string"}, ["recurring"]),
]


def build_api_tools() -> list[Tool]:
    return [
        Tool(n, d, {k: {"type": t, "required": True} for k, t in p.items()}, _mock(n), tags)
        for n, d, p, tags in _CATALOG
    ]


def load_postman(path: str) -> list[Tool]:
    def walk(items):
        for it in items:
            if "item" in it:
                yield from walk(it["item"])
            else:
                yield it

    def params_of(req):
        url = req.get("url", {})
        url = url if isinstance(url, dict) else {}
        names = [q["key"] for q in url.get("query", []) if q.get("key")]
        names += [v["key"] for v in url.get("variable", []) if v.get("key")]
        raw = (req.get("body") or {}).get("raw", "")
        try:
            body = json.loads(raw) if raw else {}
            names += list(body) if isinstance(body, dict) else []
        except ValueError:
            pass
        return {n: {"type": "string", "required": False} for n in dict.fromkeys(names)}

    data = json.load(open(path, encoding="utf-8"))
    tools = []
    for it in walk(data.get("item", [])):
        req = it.get("request", {})
        name = "".join(c if c.isalnum() else "_" for c in it["name"]).lower()
        desc = req.get("description", "") or it["name"]
        if isinstance(desc, dict):
            desc = desc.get("content", "")
        tools.append(Tool(name, str(desc)[:300], params_of(req), _mock(name)))
    return tools


# RAG tool: knowledge base lookup
KB = [
    "Invoices can be cancelled only while they are in SENT or UNPAID state.",
    "Disputes must be responded to within 10 days or the claim is lost.",
    "Refunds are possible within 180 days of the original payment.",
    "Payouts require the recipient to have a verified PayPal email.",
]


def rag_search(query: str) -> dict:
    r = Retriever(KB)
    return {"passages": [KB[i] for i, _ in r.search(query, 2)]}


# System search tool: capabilities and request history
def make_system_search(registry_search):
    def system_search(query: str) -> dict:
        q = query.lower()
        if any(w in q for w in ("status", "last request", "history", "previous")):
            return {"recent_requests": REQUEST_LOG[-3:]}
        return {"matching_tools": [t.name + ": " + t.description for t in registry_search(query, 5)]}
    return system_search
