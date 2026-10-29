import json

import pytest

from tools import calculate_total, run_tool


def test_calculate_total_matches_example():
    assert calculate_total(850, 3, 0.10) == pytest.approx(2805)
    assert calculate_total(850, 5, 0.10) == pytest.approx(4675)


def test_run_tool_returns_rounded_json_result():
    content, is_error = run_tool("calculate_total", {"price": 850, "quantity": 3, "tax_rate": 0.1})
    assert not is_error
    assert json.loads(content) == {"price": 850.0, "quantity": 3, "tax_rate": 0.1, "total": 2805.0}


def test_whole_number_float_quantity_is_accepted():
    content, is_error = run_tool("calculate_total", {"price": 10, "quantity": 2.0, "tax_rate": 0})
    assert not is_error
    assert json.loads(content)["quantity"] == 2


@pytest.mark.parametrize(
    "args, expected_fragment",
    [
        ({"price": 850, "quantity": 3}, "Missing argument(s): tax_rate"),
        ({"price": 850, "quantity": 3, "tax_rate": 0.1, "discount": 5}, "Unexpected argument(s): discount"),
        ({"price": "850", "quantity": 3, "tax_rate": 0.1}, "'price' must be a number"),
        ({"price": 850, "quantity": True, "tax_rate": 0.1}, "'quantity' must be a number"),
        ({"price": 850, "quantity": 2.5, "tax_rate": 0.1}, "'quantity' must be a whole number"),
        ({"price": -1, "quantity": 3, "tax_rate": 0.1}, "'price' cannot be negative"),
        ({"price": 850, "quantity": -3, "tax_rate": 0.1}, "'quantity' cannot be negative"),
        ({"price": 850, "quantity": 3, "tax_rate": 10}, "'tax_rate' must be a decimal between 0 and 1"),
        ({"price": float("inf"), "quantity": 3, "tax_rate": 0.1}, "must be a finite number"),
        ("not an object", "Arguments must be a JSON object"),
    ],
)
def test_invalid_arguments_return_error_not_exception(args, expected_fragment):
    content, is_error = run_tool("calculate_total", args)
    assert is_error
    assert expected_fragment in content


def test_unknown_tool_returns_error():
    content, is_error = run_tool("delete_everything", {})
    assert is_error
    assert "unknown tool" in content


def test_tool_that_raises_returns_error(monkeypatch):
    import tools

    def boom(**kwargs):
        raise RuntimeError("database offline")

    monkeypatch.setattr(tools, "calculate_total", boom)
    content, is_error = run_tool("calculate_total", {"price": 1, "quantity": 1, "tax_rate": 0})
    assert is_error
    assert "RuntimeError: database offline" in content


def test_overflowing_result_returns_error():
    content, is_error = run_tool("calculate_total", {"price": 1e308, "quantity": 10, "tax_rate": 0.5})
    assert is_error
    assert "finite" in content
