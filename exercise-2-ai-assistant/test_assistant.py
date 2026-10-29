import json
from types import SimpleNamespace

import pytest

from assistant import Assistant, AssistantError


def text(value):
    return SimpleNamespace(type="text", text=value)


def tool_use(args, id="toolu_1", name="calculate_total"):
    return SimpleNamespace(type="tool_use", id=id, name=name, input=args)


def response(stop_reason, *content):
    return SimpleNamespace(stop_reason=stop_reason, content=list(content))


class FakeClient:
    def __init__(self, *responses):
        self._responses = list(responses)
        self.calls = []
        self.beta = SimpleNamespace(messages=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.calls.append(list(kwargs["messages"]))
        result = self._responses.pop(0)
        if isinstance(result, Exception):
            raise result
        return result


def last_tool_result(client, call_index):
    return client.calls[call_index][-1]["content"][0]


def test_tool_call_result_is_returned_through_llm():
    client = FakeClient(
        response("tool_use", tool_use({"price": 850, "quantity": 3, "tax_rate": 0.1})),
        response("end_turn", text("The total is $2,805.")),
    )
    assistant = Assistant(client=client)

    assert assistant.ask("3 laptops at $850 with 10% tax?") == "The total is $2,805."

    result = last_tool_result(client, 1)
    assert result["is_error"] is False
    assert json.loads(result["content"])["total"] == 2805.0


def test_no_tool_call_when_not_needed():
    client = FakeClient(response("end_turn", text("Hello!")))
    assert Assistant(client=client).ask("hi") == "Hello!"
    assert len(client.calls) == 1


def test_conversation_history_carries_across_turns():
    client = FakeClient(
        response("tool_use", tool_use({"price": 850, "quantity": 3, "tax_rate": 0.1})),
        response("end_turn", text("The total is $2,805.")),
        response("tool_use", tool_use({"price": 850, "quantity": 5, "tax_rate": 0.1}, id="toolu_2")),
        response("end_turn", text("The total for 5 laptops is $4,675.")),
    )
    assistant = Assistant(client=client)
    assistant.ask("3 laptops at $850 with 10% tax?")

    assert assistant.ask("What about 5 laptops?") == "The total for 5 laptops is $4,675."

    second_turn_request = client.calls[2]
    assert [m["role"] for m in second_turn_request] == ["user", "assistant", "user", "assistant", "user"]
    assert second_turn_request[-1]["content"] == "What about 5 laptops?"
    assert json.loads(last_tool_result(client, 3)["content"])["total"] == 4675.0


def test_invalid_arguments_are_reported_to_llm_which_can_retry():
    client = FakeClient(
        response("tool_use", tool_use({"price": 850, "quantity": 3, "tax_rate": 10})),
        response("tool_use", tool_use({"price": 850, "quantity": 3, "tax_rate": 0.1}, id="toolu_2")),
        response("end_turn", text("The total is $2,805.")),
    )
    seen = []
    assistant = Assistant(client=client, on_tool_call=lambda *call: seen.append(call))

    assert assistant.ask("3 laptops at $850 with 10% tax?") == "The total is $2,805."

    bad_result = last_tool_result(client, 1)
    assert bad_result["is_error"] is True
    assert "tax_rate" in bad_result["content"]
    assert [is_error for *_, is_error in seen] == [True, False]


def test_multiple_tool_calls_return_results_in_one_message():
    client = FakeClient(
        response(
            "tool_use",
            tool_use({"price": 850, "quantity": 3, "tax_rate": 0.1}, id="a"),
            tool_use({"price": 850, "quantity": 5, "tax_rate": 0.1}, id="b"),
        ),
        response("end_turn", text("$2,805 and $4,675.")),
    )
    Assistant(client=client).ask("3 and 5 laptops?")

    results = client.calls[1][-1]["content"]
    assert [r["tool_use_id"] for r in results] == ["a", "b"]


def test_runaway_tool_loop_is_stopped_and_history_rolled_back():
    looping = [response("tool_use", tool_use({"price": 1, "quantity": 1, "tax_rate": 0})) for _ in range(3)]
    assistant = Assistant(client=FakeClient(*looping), max_tool_rounds=2)

    with pytest.raises(AssistantError, match="2 rounds"):
        assistant.ask("loop forever")
    assert assistant.messages == []


def test_api_failure_rolls_back_the_turn():
    client = FakeClient(
        response("end_turn", text("Hello!")),
        RuntimeError("network down"),
    )
    assistant = Assistant(client=client)
    assistant.ask("hi")

    with pytest.raises(RuntimeError):
        assistant.ask("second message")
    assert [m["role"] for m in assistant.messages] == ["user", "assistant"]


def test_refusal_raises_and_rolls_back():
    assistant = Assistant(client=FakeClient(response("refusal")))
    with pytest.raises(AssistantError, match="declined"):
        assistant.ask("something disallowed")
    assert assistant.messages == []
