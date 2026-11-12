from agent.agent import Agent

if __name__ == "__main__":
    agent = Agent()
    print("PayPal agent. Type 'exit' to quit.")
    while (msg := input("you> ").strip()) != "exit":
        print("agent>", agent.run(msg))
