"""Terminal entry point: python main.py [--verbose]"""

import argparse
import json
import os
import sys

import anthropic

from assistant import Assistant, AssistantError

EXIT_COMMANDS = {"exit", "quit", "q"}


def print_tool_call(name, args, content, is_error):
    status = "error" if is_error else "ok"
    print(f"  [tool:{status}] {name}({json.dumps(args)}) -> {content}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Terminal AI assistant with tool calling.")
    parser.add_argument("--verbose", "-v", action="store_true", help="show tool calls and results")
    args = parser.parse_args()

    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("Set the ANTHROPIC_API_KEY environment variable first (see README).", file=sys.stderr)
        return 1

    assistant = Assistant(on_tool_call=print_tool_call if args.verbose else None)
    print(f"Assistant ready ({assistant.model}). Type 'exit' to quit.\n")

    while True:
        try:
            user_text = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0

        if not user_text:
            continue
        if user_text.lower() in EXIT_COMMANDS:
            return 0

        try:
            reply = assistant.ask(user_text)
        except AssistantError as exc:
            print(f"Assistant: Sorry, {exc} Please try again.\n")
            continue
        except anthropic.AuthenticationError:
            print("Error: the API key was rejected. Check ANTHROPIC_API_KEY.", file=sys.stderr)
            return 1
        except anthropic.RateLimitError:
            print("Error: rate limited by the API. Wait a moment and try again.\n")
            continue
        except anthropic.APIConnectionError:
            print("Error: could not reach the API. Check your network and try again.\n")
            continue
        except anthropic.APIStatusError as exc:
            print(f"Error: API returned {exc.status_code}: {exc.message}\n")
            continue
        except KeyboardInterrupt:
            print("\n(cancelled)\n")
            continue

        print(f"Assistant: {reply}\n")


if __name__ == "__main__":
    sys.exit(main())
