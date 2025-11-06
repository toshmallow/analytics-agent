"""CLI interface for the analytics agent."""

import sys

from analytics_agent.container import container


def main() -> None:
    """Run the analytics agent in interactive mode."""
    config = container.config()

    try:
        config.validate()
    except ValueError as e:
        print(f"Configuration error: {e}")
        print("\nPlease set up your .env file with required configuration:")
        print("  - GCP_PROJECT_ID")
        print("  - GEMINI_API_KEY")
        print("  - GOOGLE_APPLICATION_CREDENTIALS (optional)")
        sys.exit(1)

    agent = container.analytics_agent()
    tools = agent.tools

    print("Analytics Agent initialized successfully!")
    print("Ask me questions about your BigQuery data.\n")
    print("Type 'exit' or 'quit' to end the session.")
    print("Type 'reset' to clear conversation history.\n")

    while True:
        try:
            question = input("> ").strip()

            if question.lower() in ["exit", "quit"]:
                print("Goodbye!")
                break

            if question.lower() == "reset":
                agent.reset_conversation()
                print("Conversation history cleared.\n")
                continue

            if not question:
                continue

            print("\n🤖: ", end="", flush=True)

            final_response = None
            for event in agent.analyze(question):
                for node_name, node_output in event.items():
                    if node_name == "agent":
                        messages = node_output.get("messages", [])
                        if messages:
                            last_message = messages[-1]
                            if hasattr(last_message, "tool_calls") and last_message.tool_calls:
                                if last_message.content:
                                    print(f"\n{last_message.content}")

                                print("\n[Executing tools...]")
                                for tool_call in last_message.tool_calls:
                                    tool_name = tool_call.get("name", "unknown")
                                    tool_args = tool_call.get("args", {})

                                    formatted_call = None
                                    for tool in tools:
                                        if tool_name in tool.get_tool_names():
                                            formatted_call = tool.format_tool_call(
                                                tool_name, tool_args
                                            )
                                            break

                                    if formatted_call:
                                        print(f"\n{formatted_call}")
                            elif last_message.content:
                                final_response = last_message.content

                    elif node_name == "tools":
                        messages = node_output.get("messages", [])
                        for msg in messages:
                            if hasattr(msg, "content") and hasattr(msg, "name"):
                                tool_name = msg.name
                                result_content = str(msg.content)

                                formatted_result = None
                                for tool in tools:
                                    if tool_name in tool.get_tool_names():
                                        formatted_result = tool.format_tool_result(
                                            tool_name, result_content
                                        )
                                        break

                                if formatted_result:
                                    print(f"\n{formatted_result}")

            if final_response:
                print("\n✅ Final response received")
                print(f"\n\n{final_response}")
            print()

        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"\nError: {e}")
            print("Please try again.\n")


if __name__ == "__main__":
    main()
