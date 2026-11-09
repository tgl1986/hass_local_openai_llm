# ruff: noqa: F401, F811
# Imported native fixture is registered by pytest and requested by name.
"""Native tool events use HA's external-call contract without executing tools."""

import asyncio
from copy import deepcopy
from unittest.mock import AsyncMock

import pytest
from homeassistant.components import conversation

from custom_components.local_openai.entity import GenerationMetadata, WebUIChat
from test_webui import chatlog, events, native


def call(call_id="weather", name="get_weather", arguments='{"city":"Berlin"}'):
    return {
        "type": "function_call",
        "id": "item_" + call_id,
        "call_id": call_id,
        "name": name,
        "arguments": arguments,
        "status": "completed",
    }


def result(call_id="weather", output=None):
    return {
        "type": "function_call_output",
        "id": "out_" + call_id,
        "call_id": call_id,
        "output": output
        if output is not None
        else [{"type": "input_text", "text": "sonnig"}],
        "status": "completed",
    }


def done_item(item):
    return {
        "type": "response:completion",
        "data": {"type": "response.output_item.done", "item": item},
    }


def text(value, item_id="answer"):
    return {
        "type": "response:completion",
        "data": {
            "type": "response.output_text.delta",
            "item_id": item_id,
            "delta": value,
        },
    }


def snapshot(items, done=False):
    return {"type": "chat:completion", "data": {"output": items, "done": done}}


async def consume(native, log, state):
    async for _ in log.async_add_delta_content_stream(
        native.entity_id, native._async_webui_receive(state, GenerationMetadata())
    ):
        pass


async def test_announcement_call_result_answer_are_ordered_external_and_not_duplicated(
    native, hass
):
    state, log = WebUIChat(), chatlog(hass)
    # With no HA LLM API, any accidental non-external call is rejected by real ChatLog.
    assert log.llm_api is None
    deltas = []
    log.delta_listener = lambda _log, delta: deltas.append(delta)
    task = asyncio.create_task(consume(native, log, state))
    state.events.put_nowait(text("Ich prüfe das Wetter. ", "announcement"))
    await asyncio.sleep(0)
    assert any(d.get("content") == "Ich prüfe das Wetter. " for d in deltas)
    assert not task.done()
    state.events.put_nowait(done_item(call()))
    state.events.put_nowait(done_item(call()))
    await asyncio.sleep(0)
    tools = [t for delta in deltas for t in delta.get("tool_calls", [])]
    assert len(tools) == 1
    assert tools[0].external is True
    assert not task.done()
    state.events.put_nowait(done_item(result()))
    state.events.put_nowait(text("In Berlin ist es sonnig."))
    final_items = [
        {
            "type": "message",
            "id": "announcement",
            "content": [{"text": "Ich prüfe das Wetter. "}],
        },
        call(),
        result(),
        {
            "type": "message",
            "id": "answer",
            "content": [{"text": "In Berlin ist es sonnig."}],
        },
    ]
    state.events.put_nowait(snapshot(final_items, True))
    await task
    assert [content.role for content in log.content[2:]] == [
        "assistant",
        "tool_result",
        "assistant",
    ]
    announcement, tool_result, answer = log.content[-3:]
    assert announcement.content == "Ich prüfe das Wetter. "
    assert announcement.tool_calls[0].id == "weather"
    assert announcement.tool_calls[0].tool_args == {"city": "Berlin"}
    assert tool_result.tool_name == "get_weather"
    assert tool_result.result.data == {"output": result()["output"]}
    assert answer.content == "In Berlin ist es sonnig."
    assert (
        "".join(d.get("content", "") for d in deltas)
        == announcement.content + answer.content
    )
    assert len([d for d in deltas if d.get("role") == "tool_result"]) == 1


async def test_snapshot_only_parallel_and_second_round_preserve_order(native, hass):
    state, log = WebUIChat(), chatlog(hass)
    payload = [
        {"type": "input_text", "text": '{"temperature":12}'},
        {"type": "input_image", "image_url": "fixture://image"},
    ]
    items = [
        call("one"),
        call("two", "search_tools"),
        result("two", payload),
        result("one"),
        {
            "type": "message",
            "id": "between",
            "content": [{"text": "Ich prüfe auch morgen. "}],
        },
        call("three"),
        result("three"),
        {"type": "message", "id": "final", "content": [{"text": "Fertig."}]},
    ]
    untouched = deepcopy(items)
    state.events.put_nowait(snapshot(items))
    state.events.put_nowait(snapshot(items, True))
    await consume(native, log, state)
    assert items == untouched
    calls = [
        tool
        for content in log.content
        if isinstance(content, conversation.AssistantContent)
        for tool in content.tool_calls or []
    ]
    assert [tool.id for tool in calls] == ["one", "two", "three"]
    assert all(tool.external for tool in calls)
    results = [
        content
        for content in log.content
        if isinstance(content, conversation.ToolResultContent)
    ]
    assert [content.tool_call_id for content in results] == ["two", "one", "three"]
    assert results[0].result.data == {"output": payload}
    assert log.content[-1].content == "Fertig."


async def test_pending_then_resume_does_not_repeat_calls_or_questions(native, hass):
    state, log = WebUIChat(), chatlog(hass)
    for event in events("question"):
        state.events.put_nowait(event)
    await consume(native, log, state)
    question = log.content[-1].content
    assert "Für welche Stadt" in question
    assert state.pending and len(state.tool_calls) == 1 and not state.tool_results
    log.async_add_user_content(conversation.UserContent(content="München"))
    state.pending = None
    await consume(native, log, state)
    calls = [
        tool
        for content in log.content
        if isinstance(content, conversation.AssistantContent)
        for tool in content.tool_calls or []
    ]
    results = [
        content
        for content in log.content
        if isinstance(content, conversation.ToolResultContent)
    ]
    assert len(calls) == len(results) == 1
    assert results[0].tool_call_id == calls[0].id == "call_voice_city"
    assert (
        log.content[-1].content
        == "Ich plane den Spaziergang für München. Die Rückfrage ist beantwortet."
    )


@pytest.mark.parametrize("arguments", ["{invalid", "", {"city": "Berlin"}])
async def test_provider_arguments_are_reported_without_local_execution(
    native, hass, arguments
):
    state, log = WebUIChat(), chatlog(hass)
    state.events.put_nowait(done_item(call(arguments=arguments)))
    state.events.put_nowait(
        done_item(result(output=[{"type": "input_text", "text": "provider result"}]))
    )
    state.events.put_nowait(text("Antwort des Modells."))
    state.events.put_nowait(snapshot([], True))
    await consume(native, log, state)
    tool = log.content[-3].tool_calls[0]
    assert tool.external
    assert log.content[-1].content == "Antwort des Modells."


async def test_new_normal_turn_resets_tool_event_positions(native):
    state = WebUIChat(tool_calls={"weather": "get_weather"}, tool_results={"weather"})
    native._async_webui_request = AsyncMock(
        side_effect=[
            {
                "data": [
                    {
                        "id": "test-model",
                        "name": "test-model",
                        "info": {"meta": {}, "params": {}},
                    }
                ]
            },
            {"id": "new-chat"},
        ]
    )
    from unittest.mock import MagicMock

    native._webui_socket = MagicMock()
    await native._async_webui_prepare_turn(
        state,
        {
            "model": "test-model",
            "messages": [{"role": "user", "content": "Noch einmal."}],
        },
        {},
        None,
    )
    assert not state.tool_calls and not state.tool_results


async def test_final_result_without_followup_text_has_ha_silent_assistant(native, hass):
    state, log = WebUIChat(), chatlog(hass)
    deltas = []
    log.delta_listener = lambda _log, delta: deltas.append(delta)
    state.events.put_nowait(snapshot([call(), result()], True))
    await consume(native, log, state)
    assert isinstance(log.content[-1], conversation.AssistantContent)
    assert log.content[-1].content == ""
    assert [delta.get("content") for delta in deltas if "content" in delta] == [""]
