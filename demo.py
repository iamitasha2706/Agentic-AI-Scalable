import sys

from agent.agent import Agent

MESSAGES = [
    "Send an invoice for $50 to bob@x.com",
    "What was my total sales volume last month?",
    "Is there a dispute open from user_123?",
    "What tools are available for managing invoices?",
    "Can I cancel an invoice? What is the policy",
    "What is the status of my last request?",
    "Send an invoice for $75",
    "bob@x.com",
]


class BrokenLLM:
    def decide(self, *args, **kwargs):
        raise TimeoutError("simulated")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    agent = Agent()
    for m in MESSAGES:
        print(m, "->", agent.run(m))
    print("broken ->", Agent(llm=BrokenLLM()).run("hello"))
