# Exercise 2: Terminal AI Assistant with Tool Calling

A terminal chat assistant that lets the LLM decide when to call the provided
`calculate_total(price, quantity, tax_rate)` function. Python runs the
function, and the LLM phrases the answer.

```
You: What is the total price of 3 laptops costing $850 each, including 10% tax?
Assistant: The total is $2,805.

You: What about 5 laptops?
Assistant: The total for 5 laptops is $4,675.
```

## Run

```bash
cd exercise-2-ai-assistant
pip install -r requirements.txt

# Provide the API key as an environment variable (never commit it)
export ANTHROPIC_API_KEY=...          # macOS / Linux
$env:ANTHROPIC_API_KEY = "..."        # Windows PowerShell

python main.py            # chat
python main.py --verbose  # also print each tool call and its result
python -m pytest -q       # tests; these need no API key and make no network calls
```

Optional: set `ANTHROPIC_MODEL` to use a different Claude model. The default is `claude-opus-5-5`.
Type `exit`, `quit`, or press Ctrl+C to leave.

## Design

| File | Responsibility |
| --- | --- |
| `tools.py` | The provided `calculate_total` (unchanged), its JSON schema for the LLM, argument validation, and `run_tool()`, which never raises. |
| `assistant.py` | `Assistant`: conversation history plus the hand-written tool-use loop. |
| `main.py` | The terminal REPL and user-facing error messages. |

### How a turn works

1. The user message is appended to the history, and the history is sent with the tool definitions.
2. The **LLM decides** whether to call `calculate_total` (`tool_choice` is left at `auto`).
3. If it calls the tool (`stop_reason == "tool_use"`), each call is validated and run
   locally. The result goes back as a `tool_result` block, and the loop continues.
4. When the LLM answers in text, that answer is returned to the user.

The whole history, including tool calls and results, is kept between turns. That
is how *"What about 5 laptops?"* reuses the $850 price and 10% tax from the first turn.

### Error handling

| Failure | Behaviour |
| --- | --- |
| Invalid tool arguments (wrong type, missing or extra field, negative values, `quantity` not a whole number, `tax_rate` outside 0–1 such as `10` for 10%) | Returned to the LLM as `tool_result` with `is_error: true` and a specific message, so it can correct the arguments and retry. |
| Tool raises an exception, or the result is not finite | Caught and returned to the LLM as an error result. The chat does not crash. |
| Unknown tool name | Error result listing the available tools. |
| LLM keeps calling tools without answering | Stopped after 5 rounds. |
| API error (network, rate limit, auth), refusal, or truncated response | The turn's messages are **rolled back**, so the history stays valid. The user sees a clear message and can retry. |

Validation lives in Python rather than relying only on the JSON schema. The schema
guides the model, but the code must never trust model output.

## Assumptions and choices

- **Provider:** Anthropic Claude, called through the official `anthropic` Python SDK.
  There is no LangChain, LangGraph or other agent framework; the loop is about 40
  lines in `assistant.py`.
- **`tax_rate` is a decimal fraction** (10% → `0.10`), matching the provided
  function. The system prompt and tool description tell the model this. Values
  above 1 are rejected, so a mistaken `10` cannot produce a total 11 times too large.
- **Money is rounded to cents** in the tool result. The LLM formats it for display.
- **Effort is set to `low`.** This is simple chat, and low effort keeps replies quick.
- **Refusal fallback:** the request opts into Anthropic's server-side
  `fallbacks: "default"` (beta), so a false-positive safety decline is retried
  on a recommended model instead of failing.
- **The API key comes from `ANTHROPIC_API_KEY` only.** `.env` files are git-ignored,
  and `.env.example` lists the variables without values.
- **History is kept in memory** for the session. Persistence was not required.
