import sys

sys.stdout.reconfigure(encoding="utf-8")

from agent.agent import Agent

a = Agent()
for m in [
    "Send an invoice for $50 to bob@x.com",
    "What was my total sales volume last month?",
    "Is there a dispute open from user_123?",
    "What tools are available for managing invoices?",
    "Can I cancel an invoice? What is the policy",
    "What is the status of my last request?",
    "Send an invoice for $75",
    "bob@x.com",
]:
    print(m, "->", a.run(m))

class BrokenLLM:
    def decide(self, *a, **k):
        raise TimeoutError("simulated")

print("broken ->", Agent(llm=BrokenLLM()).run("hello"))
