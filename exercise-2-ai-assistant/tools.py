"""Tools the assistant can call, their JSON schemas, and a safe dispatcher.

The LLM never runs code itself. It asks for a tool by name with JSON
arguments, and `run_tool` validates those arguments, runs the matching Python
function, and returns a result string plus an error flag. Every failure
becomes an error message the model can read and recover from, never an
exception that crashes the chat.
"""

import json
import math
from typing import Any, Callable


def calculate_total(price: float, quantity: int, tax_rate: float) -> float:
    # Provided function, unchanged.
    return price * quantity * (1 + tax_rate)


class ToolArgumentError(ValueError):
    """Raised when the model supplies arguments a tool cannot accept."""


def _number(args: dict, name: str) -> float:
    value = args.get(name)
    # bool is a subclass of int in Python; reject it explicitly.
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ToolArgumentError(f"'{name}' must be a number, got {value!r}.")
    if not math.isfinite(value):
        raise ToolArgumentError(f"'{name}' must be a finite number, got {value!r}.")
    return float(value)


def validate_calculate_total_args(args: Any) -> dict:
    """Check types and ranges, and return clean keyword arguments."""
    if not isinstance(args, dict):
        raise ToolArgumentError("Arguments must be a JSON object.")

    expected = {"price", "quantity", "tax_rate"}
    missing = expected - args.keys()
    unexpected = args.keys() - expected
    if missing:
        raise ToolArgumentError(f"Missing argument(s): {', '.join(sorted(missing))}.")
    if unexpected:
        raise ToolArgumentError(f"Unexpected argument(s): {', '.join(sorted(unexpected))}.")

    price = _number(args, "price")
    if price < 0:
        raise ToolArgumentError("'price' cannot be negative.")

    quantity = _number(args, "quantity")
    if not quantity.is_integer():
        raise ToolArgumentError(f"'quantity' must be a whole number, got {args['quantity']!r}.")
    if quantity < 0:
        raise ToolArgumentError("'quantity' cannot be negative.")

    tax_rate = _number(args, "tax_rate")
    if not 0 <= tax_rate <= 1:
        raise ToolArgumentError(
            f"'tax_rate' must be a decimal between 0 and 1 (e.g. 0.10 for 10%), got {tax_rate!r}."
        )

    return {"price": price, "quantity": int(quantity), "tax_rate": tax_rate}


def _run_calculate_total(args: Any) -> str:
    clean = validate_calculate_total_args(args)
    total = calculate_total(**clean)
    if not math.isfinite(total):
        raise ArithmeticError("The calculation did not produce a finite number.")
    return json.dumps({**clean, "total": round(total, 2)})


TOOL_DEFINITIONS = [
    {
        "name": "calculate_total",
        "description": (
            "Calculate the total price of buying `quantity` items at `price` each, "
            "including sales tax. Use this whenever the user asks for a total cost, "
            "rather than doing the arithmetic yourself."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "price": {
                    "type": "number",
                    "description": "Unit price of one item, in dollars. Must be >= 0.",
                },
                "quantity": {
                    "type": "integer",
                    "description": "Number of items. A whole number >= 0.",
                },
                "tax_rate": {
                    "type": "number",
                    "description": "Tax rate as a decimal fraction: 10% is 0.10. Between 0 and 1.",
                },
            },
            "required": ["price", "quantity", "tax_rate"],
            "additionalProperties": False,
        },
    }
]

_HANDLERS: dict[str, Callable[[Any], str]] = {
    "calculate_total": _run_calculate_total,
}


def run_tool(name: str, args: Any) -> tuple[str, bool]:
    """Run a tool by name. Returns (content, is_error) and never raises."""
    handler = _HANDLERS.get(name)
    if handler is None:
        return f"Error: unknown tool '{name}'. Available tools: {', '.join(_HANDLERS)}.", True
    try:
        return handler(args), False
    except ToolArgumentError as exc:
        return f"Error: invalid arguments for {name}: {exc}", True
    except Exception as exc:  # A failing tool must not crash the chat.
        return f"Error: {name} failed: {type(exc).__name__}: {exc}", True
