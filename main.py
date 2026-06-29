from orchestrator import MultiAgentOrchestrator

def main():
    bot     = MultiAgentOrchestrator()
    session = {}

    print("Saffron Table Bistro — Assistant")
    print("Type 'quit' to exit | 'reset' to clear session\n")
    print("=" * 50)

    while True:
        user_input = input("\nYou: ").strip()

        if not user_input:
            continue
        if user_input.lower() == "quit":
            print("Goodbye!")
            break
        if user_input.lower() == "reset":
            session = {}
            print("Session reset.")
            continue

        response = bot.handle(user_input, session)
        print(f"\nAssistant: {response}")


if __name__ == "__main__":
    main()
