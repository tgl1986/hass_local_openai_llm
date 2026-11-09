"""Native WebUI contracts: recorded wire data and the real HA ChatLog.

HTTP/socket boundaries are mocked unless explicitly marked as live validation.
The recordings originate from the preserved sandbox research, not generated SSE.
"""

from __future__ import annotations

import asyncio
from builtins import anext
import json
from copy import deepcopy
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import aiohttp
import pytest
from homeassistant.components import conversation
from homeassistant.components.conversation.const import ChatLogEventType
from homeassistant.exceptions import HomeAssistantError

from custom_components.local_openai.config_flow import ConversationFlowHandler
from custom_components.local_openai.const import CONF_OPENWEBUI_NATIVE
from custom_components.local_openai.entity import (
    GenerationMetadata,
    LocalAiEntity,
    WebUIChat,
)

FIXTURES = Path(__file__).parents[1] / "fixtures"
FINAL = "Ich plane den Spaziergang für München. Die Rückfrage ist beantwortet."


def events(name):
    return json.loads(
        (FIXTURES / f"webui_{name}_events.json").read_text(encoding="utf-8-sig")
    )


def pending_call():
    return next(
        deepcopy(item)
        for event in events("question")
        if isinstance(event.get("data"), dict)
        for item in event["data"].get("output", [])
        if item.get("status") == "pending"
    )


@pytest.fixture
def native(mock_conversation_entity):
    entity = mock_conversation_entity
    object.__setattr__(
        entity.subentry, "data", {"model": "test-model", CONF_OPENWEBUI_NATIVE: True}
    )
    entity.entity_id = "conversation.webui_test"
    entity.entry.runtime_data.base_url = "http://webui.invalid/api/"
    entity.entry.runtime_data.api_key = "fixture-token"
    return entity


def chatlog(hass, text="Plane einen Spaziergang"):
    log = conversation.ChatLog(hass, "ha-conversation")
    log.async_add_user_content(conversation.UserContent(content=text))
    return log


async def test_recorded_scalar_and_snapshot_are_not_duplicated(native):
    state = WebUIChat()
    for event in events("normal"):
        state.events.put_nowait(event)
    chunks = [
        chunk
        async for chunk in native._async_webui_receive(state, GenerationMetadata())
    ]
    expected = (
        "FINAL native_saved_chat_637d088126: EXECUTED_native_saved_chat_637d088126"
    )
    assert "".join(c.get("content", "") for c in chunks) == expected
    assert state.pending is None


async def test_recorded_pending_and_resume_use_real_ha_chatlog(native, hass):
    state = WebUIChat(chat_id="saved", assistant_id="same-assistant")
    log = chatlog(hass)
    for event in events("question"):
        state.events.put_nowait(event)
    async for _ in log.async_add_delta_content_stream(
        native.entity_id, native._async_webui_receive(state, GenerationMetadata())
    ):
        pass
    assert "Für welche Stadt" in log.content[-1].content
    assert "Berlin" in log.content[-1].content
    assert state.pending["call_id"] == "call_voice_city"
    assert state.events.qsize() > 0
    assert native._webui_answer(state, "München") == {
        "call_id": "call_voice_city",
        "action": "answer",
        "answers": {"city": "München"},
    }
    state.pending = None
    state.answers.clear()
    log.async_add_user_content(conversation.UserContent(content="München"))
    async for _ in log.async_add_delta_content_stream(
        native.entity_id, native._async_webui_receive(state, GenerationMetadata())
    ):
        pass
    assert log.content[-1].content == FINAL
    assert state.assistant_id == "same-assistant"


async def test_pending_followup_resolves_same_message_without_new_completion(
    native, hass
):
    log = chatlog(hass, "München")
    state = WebUIChat(chat_id="saved", assistant_id="original", pending=pending_call())
    native._webui_chats[log.conversation_id] = state
    native._async_webui_connect = AsyncMock()
    native._async_webui_prepare_turn = AsyncMock()
    calls = []

    async def request(method, path, headers, body=None):
        calls.append((method, path, body))
        if method == "POST":
            recording = events("question")
            resolved = next(
                i
                for i, e in enumerate(recording)
                if isinstance(e.get("data"), dict)
                and any(
                    x.get("type") == "function_call_output"
                    for x in e["data"].get("output", [])
                )
            )
            for event in recording[resolved:]:
                state.events.put_nowait(event)
            return {"status": True, "task_ids": ["native-task"]}
        return {
            "chat": {"history": {"messages": {"original": {"model": "test-model"}}}}
        }

    native._async_webui_request = request
    await native._async_handle_chat_log(log, user_input=MagicMock(text="München"))
    assert calls[0] == (
        "POST",
        "/api/v1/chats/saved/messages/original/resolve",
        {
            "call_id": "call_voice_city",
            "action": "answer",
            "answers": {"city": "München"},
        },
    )
    # Completion metadata arrives on the native socket; no blocking final GET.
    assert len(calls) == 1
    native._async_webui_prepare_turn.assert_not_awaited()
    assert log.content[-1].content == FINAL
    assert state.pending is None


@pytest.mark.parametrize("failure", ["connect", "resolve"])
async def test_failed_resolution_can_be_retried(native, hass, failure):
    state = WebUIChat(chat_id="saved", assistant_id="original", pending=pending_call())
    log = chatlog(hass, "München")
    native._webui_chats[log.conversation_id] = state
    native._async_webui_connect = AsyncMock(
        side_effect=aiohttp.ClientConnectionError("offline")
        if failure == "connect"
        else None
    )
    native._async_webui_request = AsyncMock(
        side_effect=aiohttp.ClientConnectionError("offline")
    )
    for _ in range(2):
        with pytest.raises(HomeAssistantError, match="offline"):
            await native._async_webui_turn(log, user_input=MagicMock(text="München"))
        assert state.pending
        assert state.answers == {}


@pytest.mark.parametrize(
    ("answer", "action"), [("Ja!", "approve"), ("Nein", "reject"), ("vielleicht", None)]
)
def test_approval_requires_explicit_decision(answer, action):
    state = WebUIChat(
        pending={"name": "configured_tool", "call_id": "approval", "arguments": "{}"}
    )
    payload = LocalAiEntity._webui_answer(state, answer)
    assert payload == ({"call_id": "approval", "action": action} if action else None)


def test_multiple_questions_keep_native_ids_and_selection_rules():
    call = pending_call()
    args = json.loads(call["arguments"])
    args["allow_other"] = False
    args["questions"][0]["allow_other"] = False
    args["questions"].append({**args["questions"][0], "id": "second"})
    call["arguments"] = json.dumps(args)
    state = WebUIChat(pending=call)
    assert LocalAiEntity._webui_answer(state, "Paris") is None
    assert state.answers == {}
    assert LocalAiEntity._webui_answer(state, "1") is None
    assert state.answers == {"city": "Berlin"}
    assert LocalAiEntity._webui_answer(state, "Hamburg")["answers"] == {
        "city": "Berlin",
        "second": "Hamburg",
    }


async def test_saved_turn_preserves_workspace_options_content_and_existing_branches(
    native,
):
    state = WebUIChat(chat_id="saved", assistant_id="previous")
    original = {
        "id": "previous",
        "parentId": "old-user",
        "childrenIds": ["other-branch"],
        "role": "assistant",
        "content": "existing answer",
        "output": [{"type": "function_call_output"}],
    }
    saved = {
        "history": {"messages": {"previous": deepcopy(original)}},
        "params": {"old": "keep"},
    }
    model = {
        "id": "test-model",
        "info": {
            "meta": {
                "toolIds": ["configured-only"],
                "defaultFeatureIds": ["web_search"],
                "skillIds": ["skill"],
                "terminalId": "terminal",
                "defaultFilterIds": ["filter"],
                "capabilities": {"vision": True},
            },
            "params": {"seed": 42},
        },
    }
    native._webui_socket = MagicMock()
    native._webui_socket.get_sid.return_value = "socket-session"
    native._async_webui_request = AsyncMock(
        side_effect=[{"data": [model]}, {"chat": saved}, {"id": "saved"}]
    )
    content = [
        {"type": "text", "text": "question"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,fixture"}},
    ]
    body = await native._async_webui_prepare_turn(
        state,
        {
            "model": "test-model",
            "temperature": 0.3,
            "tools": ["do-not-send"],
            "messages": [
                {"role": "system", "content": "existing prompt"},
                {"role": "user", "content": content},
            ],
        },
        {
            "features": {"web_search": False, "memory": True},
            "files": [{"id": "attachment"}],
            "params": {"top_k": 12},
            "background_tasks": {"title_generation": True},
        },
        {"role": "assistant", "content": "existing injected context"},
    )
    posted = native._async_webui_request.call_args.args[3]["chat"]
    assert body["tool_ids"] == ["configured-only"]
    assert body["features"] == {"web_search": True}
    assert body["skill_ids"] == ["skill"] and body["filter_ids"] == ["filter"]
    assert body["terminal_id"] == "terminal"
    assert "tools" not in body and "tool_choice" not in body
    assert body["params"] == {
        "seed": 42,
        "top_k": 12,
        "temperature": 0.3,
        "function_calling": "native",
    }
    assert all(message["role"] != "system" for message in body["messages"])
    assert posted["params"]["old"] == "keep"
    assert posted["files"] == [{"id": "attachment"}]
    assert body["background_tasks"]["title_generation"] is True
    assert body["user_message"]["content"] == content
    assert body["session_id"] == "socket-session" and body["stream"] is True
    history = posted["history"]["messages"]
    previous = history["previous"]
    assert previous["parentId"] == original["parentId"]
    assert previous["content"] == original["content"]
    assert previous["output"] == original["output"]
    assert previous["childrenIds"][0] == "other-branch"
    context = history[previous["childrenIds"][1]]
    assert context["content"] == "existing injected context"
    assert history[body["user_message"]["id"]]["parentId"] == context["id"]
    assert posted["history"]["currentId"] == body["id"] == state.assistant_id
    assert posted["messages"] == list(history.values())


async def test_cancellation_forwards_native_stop(native, hass):
    state = WebUIChat(chat_id="saved", assistant_id="original", pending=pending_call())
    log = chatlog(hass, "Berlin")
    native._webui_chats[log.conversation_id] = state
    native._async_webui_connect = AsyncMock()

    async def request(method, path, headers, body=None):
        if path.endswith("/resolve"):
            state.events.put_nowait(asyncio.CancelledError())
        return {"status": True}

    native._async_webui_request = AsyncMock(side_effect=request)
    with pytest.raises(asyncio.CancelledError):
        await native._async_webui_turn(log, user_input=MagicMock(text="Berlin"))
    assert native._async_webui_request.call_args.args[1] == "/api/tasks/chat/saved/stop"
    assert native._async_webui_request.await_count == 2
    assert state.active is False


def test_ha_chat_cleanup_preserves_other_chats(native):
    state = WebUIChat(chat_id="saved", active=True)
    native._webui_chats = {"expired": state, "keep": WebUIChat(chat_id="other")}
    native._webui_chat_event("expired", ChatLogEventType.DELETED, {})
    assert list(native._webui_chats) == ["keep"]
    assert isinstance(state.events.get_nowait(), asyncio.CancelledError)


async def test_unload_attempts_every_active_stop_and_closes_socket(native):
    native._webui_chats = {
        "one": WebUIChat(chat_id="one", active=True),
        "two": WebUIChat(chat_id="two", active=True),
    }
    native._async_webui_stop = AsyncMock(
        side_effect=[HomeAssistantError("stop failed"), None]
    )
    native._webui_socket = MagicMock(disconnect=AsyncMock())
    await native.async_will_remove_from_hass()
    assert native._async_webui_stop.await_count == 2
    native._webui_socket.disconnect.assert_awaited_once()
    assert not native._webui_chats


async def test_unload_survives_chat_cleanup_and_disconnect_failure(native):
    states = {
        "one": WebUIChat(chat_id="one", active=True),
        "two": WebUIChat(chat_id="two", active=True),
    }
    native._webui_chats = dict(states)

    async def stop(state):
        native._webui_chats.pop("two", None)

    native._async_webui_stop = AsyncMock(side_effect=stop)
    native._webui_socket = MagicMock(
        disconnect=AsyncMock(side_effect=OSError("disconnect failed"))
    )
    with pytest.raises(OSError, match="disconnect failed"):
        await native.async_will_remove_from_hass()
    assert native._async_webui_stop.await_count == 2
    assert not native._webui_chats


async def test_request_cannot_add_workspace_tools_features_skills_or_servers(native):
    state = WebUIChat()
    native._webui_socket = MagicMock()
    native._webui_socket.get_sid.return_value = "native-session"
    model = {
        "id": "test-model",
        "info": {
            "meta": {
                "toolIds": [],
                "defaultFeatureIds": [],
                "capabilities": {"builtin_tools": True},
                "builtinTools": {"skills": False},
            }
        },
    }
    original = deepcopy(model)
    native._async_webui_request = AsyncMock(
        side_effect=[{"data": [model]}, {"id": "new-saved-chat"}]
    )
    body = await native._async_webui_prepare_turn(
        state,
        {
            "model": "test-model",
            "messages": [{"role": "user", "content": "hello"}],
        },
        {
            "tools": [{"type": "function"}],
            "tool_ids": ["extra"],
            "features": {"web_search": True},
            "skill_ids": ["extra-skill"],
            "filter_ids": ["extra-filter"],
            "terminal_id": "extra-terminal",
            "tool_servers": [{"url": "http://not-configured.invalid"}],
        },
        None,
    )
    assert body["tool_ids"] == []
    assert body["features"] == {}
    for key in ("tools", "tool_servers", "skill_ids", "filter_ids", "terminal_id"):
        assert key not in body
        assert key not in body["params"]
    assert model == original  # Built-in configuration remains server-owned.


async def test_pending_result_opens_next_voice_turn(native, hass):
    log = chatlog(hass)
    native._webui_chats[log.conversation_id] = WebUIChat(pending=pending_call())
    log.async_add_assistant_content_without_tools(
        conversation.AssistantContent(
            agent_id=native.entity_id, content="Welche Stadt?"
        )
    )
    native._async_handle_chat_log = AsyncMock(return_value=False)
    with patch.object(type(log), "async_provide_llm_data", AsyncMock()):
        result = await native._async_handle_message(
            MagicMock(spec=conversation.ConversationInput), log
        )
    assert result.continue_conversation is True


async def test_ui_schema_and_translations_expose_switch(hass, mock_config_entry):
    flow = ConversationFlowHandler()
    flow.hass = hass
    mock_config_entry.runtime_data.models.list = AsyncMock(
        return_value=MagicMock(data=[])
    )
    with patch.object(flow, "_get_entry", return_value=mock_config_entry):
        schema = await flow.get_schema()
    marker = next(key for key in schema.schema if str(key) == CONF_OPENWEBUI_NATIVE)
    assert marker.default() is False
    assert schema.schema[marker] is bool
    translations = (
        Path(__file__).parents[2] / "custom_components/local_openai/translations"
    )
    for language in ("en", "de"):
        steps = json.loads((translations / f"{language}.json").read_text())[
            "config_subentries"
        ]["conversation"]["step"]
        for step in ("user", "reconfigure"):
            assert steps[step]["data"][CONF_OPENWEBUI_NATIVE]


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        ("http://host/api/", "http://host"),
        ("https://host/prefix/api/v1/", "https://host/prefix"),
    ],
)
def test_native_url_keeps_reverse_proxy_prefix(native, url, expected):
    native.entry.runtime_data.base_url = url
    assert str(native._webui_base_url()).rstrip("/") == expected


async def test_progress_covers_preparation_without_polluting_native_request(
    native, hass
):
    """Exercise the real announcement interval, not a simulated audio timeout."""
    from custom_components.local_openai.entity import AgentContext

    log = chatlog(hass, "Nutzerfrage")
    prepared = asyncio.Event()
    announced = asyncio.Event()
    repeated = asyncio.Event()
    announcement_times = []
    started = asyncio.get_running_loop().time()
    release = asyncio.Event()
    original = list(log.content)
    captured = []

    def listener(_log, delta):
        if "Ich arbeite noch an deiner Anfrage." in delta.get("content", ""):
            assert (
                delta["content"]
                == "Ich arbeite noch an deiner Anfrage. Ich melde mich mit dem Ergebnis. "
            )
            announcement_times.append(asyncio.get_running_loop().time())
            announced.set()
            if len(announcement_times) == 2:
                repeated.set()

    log.delta_listener = listener

    async def prepare(_log, contents, *args, **kwargs):
        prepared.set()
        await release.wait()
        captured.extend(contents)
        return (
            {"model": "test-model", "extra_headers": {}},
            AgentContext(conversation_id=log.conversation_id),
            None,
        )

    native._async_prepare_chat_log = prepare
    native._async_webui_connect = AsyncMock()
    native._async_webui_prepare_turn = AsyncMock(return_value={})
    native._async_webui_request = AsyncMock(return_value={"status": True})

    async def receive(state, meta):
        yield {"role": "assistant", "content": "Vollständige Antwort."}

    native._async_webui_receive = receive
    task = asyncio.create_task(native._async_handle_chat_log(log))
    try:
        await asyncio.wait_for(prepared.wait(), 2)
        # This guard bounds this unit test; it is not an HTTP client timeout.
        await asyncio.wait_for(announced.wait(), 11)
        assert 9.8 <= announcement_times[0] - started < 11
        await asyncio.wait_for(repeated.wait(), 16)
        assert 14.8 <= announcement_times[1] - announcement_times[0] < 16
        assert not task.done()
        native._async_webui_connect.assert_not_awaited()
        release.set()
        await asyncio.wait_for(task, 2)
        assert captured == original
        assert all(
            "Ich arbeite noch an deiner Anfrage." not in (c.content or "")
            for c in captured
        )
        assert log.content[-1].content == "Vollständige Antwort."
        native._async_webui_request.assert_awaited_once()
    finally:
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)


async def test_progress_cancellation_closes_receiver_and_stops_native_task(
    native, hass
):
    from custom_components.local_openai.entity import AgentContext

    log = chatlog(hass)
    waiting = asyncio.Event()
    closed = asyncio.Event()
    native._async_prepare_chat_log = AsyncMock(
        return_value=(
            {"model": "test-model", "extra_headers": {}},
            AgentContext(conversation_id=log.conversation_id),
            None,
        )
    )
    native._async_webui_connect = AsyncMock()

    async def prepare(state, *args):
        state.chat_id = "exact-active-chat"
        return {}

    native._async_webui_prepare_turn = prepare
    native._async_webui_request = AsyncMock(return_value={"status": True})
    native._async_webui_stop = AsyncMock()

    async def receive(state, meta):
        try:
            yield {"role": "assistant"}
            waiting.set()
            await asyncio.Event().wait()
        finally:
            closed.set()

    native._async_webui_receive = receive
    task = asyncio.create_task(native._async_handle_chat_log(log))
    await asyncio.wait_for(waiting.wait(), 2)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert closed.is_set()
    native._async_webui_stop.assert_awaited_once()
    assert native._async_webui_stop.call_args.args[0].chat_id == "exact-active-chat"
    assert not native._webui_chats[log.conversation_id].active


async def test_tool_result_starts_followup_with_fresh_think_filter(native, hass):
    state, log = WebUIChat(), chatlog(hass)
    for event in [
        {
            "type": "response:completion",
            "data": {
                "type": "response.output_text.delta",
                "item_id": "before-tool",
                "delta": "<think>Vorbereitung ohne abschließenden Tag",
            },
        },
        {
            "type": "chat:completion",
            "data": {
                "output": [
                    {
                        "id": "call-item",
                        "type": "function_call",
                        "call_id": "call",
                        "name": "get_time",
                        "arguments": "{}",
                        "status": "completed",
                    },
                    {
                        "id": "result-item",
                        "type": "function_call_output",
                        "call_id": "call",
                        "output": "12:00",
                        "status": "completed",
                    },
                ]
            },
        },
        {
            "type": "response:completion",
            "data": {
                "type": "response.output_text.delta",
                "item_id": "after-tool",
                "delta": "Es ist zwölf Uhr.",
            },
        },
        {"type": "chat:completion", "data": {"done": True}},
    ]:
        state.events.put_nowait(event)
    async for _ in log.async_add_delta_content_stream(
        native.entity_id, native._async_webui_receive(state, GenerationMetadata())
    ):
        pass
    assert log.content[-1].content == "Es ist zwölf Uhr."


@pytest.mark.parametrize(
    "before,after", [("  Gedanke  ", "Gedanke"), ("Gedanke  ", "Gedanke")]
)
async def test_native_reasoning_trim_does_not_abort_visible_answer(
    native, hass, before, after
):
    state, log = WebUIChat(), chatlog(hass)
    state.events.put_nowait(
        {
            "type": "response:completion",
            "data": {
                "type": "response.reasoning_summary_text.delta",
                "item_id": "reasoning",
                "summary_index": 0,
                "delta": before,
            },
        }
    )
    state.events.put_nowait(
        {
            "type": "chat:completion",
            "data": {
                "done": True,
                "output": [
                    {
                        "id": "reasoning",
                        "type": "reasoning",
                        "summary": [{"text": after}],
                    },
                    {
                        "id": "answer",
                        "type": "message",
                        "content": [{"text": "Fertig."}],
                    },
                ],
            },
        }
    )
    async for _ in log.async_add_delta_content_stream(
        native.entity_id, native._async_webui_receive(state, GenerationMetadata())
    ):
        pass
    assert state.text[("reasoning", 0, "thinking_content")] == after
    assert log.content[-1].content == "Fertig."


@pytest.mark.parametrize("ending", ["filtered_text", "silent", "pending"])
async def test_native_turn_returns_to_original_transformer(native, hass, ending):
    """Exercise the public turn, including the one original text processor."""
    from custom_components.local_openai.entity import AgentContext

    log = chatlog(hass)
    native.subentry.data["strip_emojis"] = True
    native._async_prepare_chat_log = AsyncMock(
        return_value=(
            {"model": "test-model", "extra_headers": {}},
            AgentContext(conversation_id=log.conversation_id),
            None,
        )
    )
    native._async_webui_connect = AsyncMock()
    native._async_webui_prepare_turn = AsyncMock(return_value={})

    async def accept(method, path, headers, body=None):
        state = native._webui_chats[log.conversation_id]
        if ending == "pending":
            incoming = events("question")
        else:
            incoming = [
                {
                    "type": "response:completion",
                    "data": {
                        "type": "response.output_text.delta",
                        "item_id": "thought",
                        "delta": "<think>Unabgeschlossener Gedanke",
                    },
                },
                {
                    "type": "chat:completion",
                    "data": {
                        "output": [
                            {
                                "id": "call-item",
                                "type": "function_call",
                                "call_id": "call",
                                "name": "get_time",
                                "arguments": "{}",
                                "status": "completed",
                            },
                            {
                                "id": "result-item",
                                "type": "function_call_output",
                                "call_id": "call",
                                "output": "12:00",
                                "status": "completed",
                            },
                        ]
                    },
                },
            ]
            if ending == "filtered_text":
                incoming.append(
                    {
                        "type": "response:completion",
                        "data": {
                            "type": "response.output_text.delta",
                            "item_id": "answer",
                            "delta": "**Es ist zwölf Uhr.** 😀",
                        },
                    }
                )
            incoming.append({"type": "chat:completion", "data": {"done": True}})
        for event in incoming:
            state.events.put_nowait(event)
        return {"status": True}

    native._async_webui_request = accept
    with patch.object(
        native, "_transform_stream", wraps=native._transform_stream
    ) as original:
        await native._async_handle_chat_log(log)
        original.assert_called_once()
    assert isinstance(log.content[-1], conversation.AssistantContent)
    if ending == "filtered_text":
        # Exactly the original behavior: emoji/think removal, no invented Markdown filter.
        assert log.content[-1].content == "**Es ist zwölf Uhr.** "
    elif ending == "silent":
        assert log.content[-1].content == ""
    else:
        assert "Für welche Stadt" in log.content[-1].content
        assert native._webui_chats[log.conversation_id].pending is not None


"""Focused regression cases appended to the existing test module for review."""


@pytest.mark.parametrize(
    "kind",
    [
        "response.output_text.done",
        "response.content_part.done",
        "response.completed",
        "response.incomplete",
    ],
)
@pytest.mark.parametrize("index", [0, 2])
def test_native_final_text_forms_share_existing_deduplication(kind, index):
    state = WebUIChat()
    item = {
        "id": "answer",
        "type": "message",
        "content": [{"text": ""}] * index + [{"text": "Fertige Antwort."}],
    }
    # An earlier delta is already spoken; the done event must add only its suffix.
    first = LocalAiEntity._webui_output_text(
        state,
        {
            "type": "response:completion",
            "data": {
                "type": "response.output_text.delta",
                "item_id": "answer",
                "content_index": index,
                "delta": "Fertige ",
            },
        },
    )
    assert first == [{"content": "Fertige "}]
    data = {"type": kind, "item_id": "answer", "content_index": index}
    if kind in ("response.completed", "response.incomplete"):
        data["response"] = {"output": [item]}
    elif kind == "response.content_part.done":
        data["part"] = {"type": "output_text", "text": "Fertige Antwort."}
    else:
        data["text"] = "Fertige Antwort."
    event = {"type": "response:completion", "data": data}
    assert LocalAiEntity._webui_output_text(state, event) == [{"content": "Antwort."}]
    assert LocalAiEntity._webui_output_text(state, event) == []
    assert (
        LocalAiEntity._webui_output_text(
            state,
            {
                "type": "chat:completion",
                "data": {
                    "done": True,
                    "output": [item],
                },
            },
        )
        == []
    )


async def test_provider_completion_does_not_end_native_turn(native, hass):
    state = WebUIChat()
    state.events.put_nowait(
        {
            "type": "response:completion",
            "data": {
                "type": "response.completed",
                "response": {
                    "output": [
                        {
                            "id": "first",
                            "type": "message",
                            "content": [{"text": "Ich prüfe. "}],
                        },
                    ]
                },
            },
        }
    )
    receiver = native._async_webui_receive(state, GenerationMetadata())
    try:
        assert await anext(receiver) == {"role": "assistant"}
        assert await anext(receiver) == {"content": "Ich prüfe. "}
        state.events.put_nowait(
            {
                "type": "chat:completion",
                "data": {
                    "done": True,
                    "output": [
                        {
                            "id": "first",
                            "type": "message",
                            "content": [{"text": "Ich prüfe. "}],
                        },
                        {
                            "id": "last",
                            "type": "message",
                            "content": [{"text": "Fertig."}],
                        },
                    ],
                },
            }
        )
        assert await anext(receiver) == {"content": "Fertig."}
        with pytest.raises(StopAsyncIteration):
            await anext(receiver)
    finally:
        await receiver.aclose()


@pytest.mark.parametrize(
    "kind,with_id",
    [
        ("response.text.done", False),
        ("response.output_text.done", False),
        ("response.content_part.done", False),
        ("response.content_part.done", True),
    ],
)
def test_done_without_identity_keeps_native_snapshot_fallback(kind, with_id):
    state = WebUIChat()
    data = {
        "type": kind,
        "output_index": 1,
        "text": "Fertig.",
        "part": {"type": "output_text", "text": "Fertig."},
    }
    if with_id:
        data["item_id"] = "answer"
    assert (
        LocalAiEntity._webui_output_text(
            state,
            {
                "type": "response:completion",
                "data": data,
            },
        )
        == []
    )
    assert state.text == {}
    assert LocalAiEntity._webui_output_text(
        state,
        {
            "type": "chat:completion",
            "data": {
                "output": [
                    {
                        "id": "answer",
                        "type": "message",
                        "content": [{"text": "Fertig."}],
                    }
                ],
            },
        },
    ) == [{"content": "Fertig."}]


@pytest.mark.parametrize("kind", ["response.text.done", "response.content_part.done"])
def test_untyped_or_reasoning_done_is_not_promoted_to_speech(kind):
    state = WebUIChat()
    data = {
        "type": kind,
        "item_id": "reasoning",
        "content_index": 0,
        "text": "Privater Gedanke",
        "part": {"type": "reasoning_text", "text": "Privater Gedanke"},
    }
    assert (
        LocalAiEntity._webui_output_text(
            state,
            {
                "type": "response:completion",
                "data": data,
            },
        )
        == []
    )
    assert state.text == {}


@pytest.mark.parametrize("message_complete", [True, False])
async def test_short_native_message_boundary_controls_wait_announcement(
    native, hass, message_complete
):
    from custom_components.local_openai.entity import AgentContext

    log = chatlog(hass)
    announced = asyncio.Event()
    seen = []

    def listener(_log, delta):
        seen.append(dict(delta))
        if "Ich arbeite noch an deiner Anfrage." in delta.get("content", ""):
            announced.set()

    log.delta_listener = listener
    native._async_prepare_chat_log = AsyncMock(
        return_value=(
            {"model": "test-model", "extra_headers": {}},
            AgentContext(conversation_id=log.conversation_id),
            None,
        )
    )
    native._async_webui_connect = AsyncMock()
    native._async_webui_prepare_turn = AsyncMock(return_value={})
    native._async_webui_stop = AsyncMock()

    async def accept(*args):
        state = native._webui_chats[log.conversation_id]
        state.events.put_nowait(
            {
                "type": "response:completion",
                "data": {
                    "type": "response.output_text.delta",
                    "item_id": "prelude",
                    "delta": "Ich suche kurz.",
                },
            }
        )
        if message_complete:
            state.events.put_nowait(
                {
                    "type": "response:completion",
                    "data": {
                        "type": "response.output_item.done",
                        "item": {
                            "id": "prelude",
                            "type": "message",
                            "status": "completed",
                            "content": [
                                {"type": "output_text", "text": "Ich suche kurz."}
                            ],
                        },
                    },
                }
            )
        return {"status": True}

    native._async_webui_request = accept
    task = asyncio.create_task(native._async_handle_chat_log(log))
    try:
        if message_complete:
            await asyncio.wait_for(announced.wait(), 11)
        else:
            with pytest.raises(TimeoutError):
                await asyncio.wait_for(announced.wait(), 10.3)
        assert not task.done()
        assert sum(d.get("content", "").count("Ich suche kurz.") for d in seen) == 1
        state = native._webui_chats[log.conversation_id]
        state.events.put_nowait({"type": "chat:completion", "data": {"done": True}})
        await asyncio.wait_for(task, 1)
    finally:
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)


async def test_empty_done_message_after_tool_keeps_explicit_silent_reply(native, hass):
    state, log = WebUIChat(), chatlog(hass)
    for event in [
        {
            "type": "chat:completion",
            "data": {
                "output": [
                    {
                        "type": "function_call",
                        "id": "call",
                        "call_id": "call",
                        "name": "test",
                        "arguments": "{}",
                    },
                    {
                        "type": "function_call_output",
                        "id": "result",
                        "call_id": "call",
                        "output": "ok",
                    },
                ]
            },
        },
        {
            "type": "response:completion",
            "data": {
                "type": "response.output_item.done",
                "item": {
                    "type": "message",
                    "id": "empty",
                    "status": "completed",
                    "content": [{"text": ""}],
                },
            },
        },
        {"type": "chat:completion", "data": {"done": True}},
    ]:
        state.events.put_nowait(event)
    async for _ in log.async_add_delta_content_stream(
        native.entity_id,
        native._transform_stream(
            native._async_webui_receive(state, GenerationMetadata()),
            strip_emojis=True,
            meta=GenerationMetadata(),
            native_deltas=True,
        ),
    ):
        pass
    assert isinstance(log.content[-1], conversation.AssistantContent)
    assert log.content[-1].content == ""
