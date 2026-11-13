"""Context placement through the real native request builder and HA prompt setup."""

import json
from copy import deepcopy
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from custom_components.local_openai.config_flow import ConversationFlowHandler
from custom_components.local_openai.const import (
    CONF_OPENWEBUI_ATTACH_HA_PROMPT,
    CONF_OPENWEBUI_NATIVE,
)
from custom_components.local_openai.entity import WebUIChat
from homeassistant.components import conversation
from homeassistant.core import Context


@pytest.fixture
def context_entity(mock_conversation_entity):
    entity = mock_conversation_entity
    entity.entity_id = "conversation.context_test"
    object.__setattr__(
        entity.subentry, "data", {"model": "test-model", CONF_OPENWEBUI_NATIVE: True}
    )
    entity._webui_socket = MagicMock()
    entity._webui_socket.get_sid.return_value = "test-session"
    return entity


@pytest.mark.parametrize("attach", [False, True])
@pytest.mark.parametrize("content_format", ["string", "text_parts", "multimodal"])
async def test_ha_prompt_only_in_new_user_content(
    context_entity, attach, content_format
):
    entity = context_entity
    entity.subentry.data[CONF_OPENWEBUI_ATTACH_HA_PROMPT] = attach
    user_content = (
        [
            {"type": "text", "text": "QUESTION"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,test"}},
        ]
        if content_format == "multimodal"
        else "QUESTION"
    )
    if content_format == "text_parts":
        # Use the actual HA-to-Completions conversion, including text-only input.
        converted = await entity._convert_content_to_chat_message(
            conversation.UserContent(content="QUESTION"), "ha-conversation"
        )
        user_content = converted["content"]
    original_user_content = deepcopy(user_content)
    model = {
        "id": "test-model",
        "info": {
            "params": {"system": "WORKSPACE_SYSTEM", "seed": 42},
            "meta": {"toolIds": ["weather"]},
        },
    }
    old_message = {
        "id": "previous",
        "parentId": None,
        "childrenIds": [],
        "role": "assistant",
        "content": "PREVIOUS_ANSWER",
    }
    saved = {
        "history": {
            "currentId": "previous",
            "messages": {"previous": deepcopy(old_message)},
        },
        "params": {"system": "OLD_HA_SYSTEM", "keep": 17},
    }
    entity._async_webui_request = AsyncMock(
        side_effect=[{"data": [model]}, {"chat": saved}, {"id": "saved"}]
    )
    body = await entity._async_webui_prepare_turn(
        WebUIChat(chat_id="saved", assistant_id="previous"),
        {
            "model": "test-model",
            "messages": [
                {"role": "system", "content": "HA_PROMPT"},
                {"role": "user", "content": user_content},
            ],
        },
        {"params": {"system": "EXTRA_SYSTEM"}, "files": [{"id": "file"}]},
        None,
    )
    expected = (
        (
            [{"type": "text", "text": "HA_PROMPT"}, *user_content]
            if content_format != "string"
            else "HA_PROMPT\n\nQUESTION"
        )
        if attach
        else user_content
    )
    posted = entity._async_webui_request.call_args.args[3]["chat"]
    saved_content = (
        ("HA_PROMPT\n\nQUESTION" if attach else "QUESTION")
        if content_format == "text_parts"
        else expected
    )
    assert body["user_message"]["content"] == saved_content
    assert body["messages"][-1]["content"] == expected
    assert (
        posted["history"]["messages"][body["user_message"]["id"]]["content"]
        == saved_content
    )
    assert (
        next(
            message
            for message in posted["messages"]
            if message["id"] == body["user_message"]["id"]
        )["content"]
        == saved_content
    )
    assert "system" not in body["params"] and "system" not in posted["params"]
    assert all(m["role"] != "system" for m in body["messages"])
    assert posted["params"]["keep"] == 17
    assert body["files"] == posted["files"] == [{"id": "file"}]
    assert (
        posted["history"]["messages"]["previous"]["content"] == old_message["content"]
    )
    assert model["info"]["params"]["system"] == "WORKSPACE_SYSTEM"
    assert body["tool_ids"] == ["weather"] and "tools" not in body
    assert user_content == original_user_content


async def test_checkbox_toggle_does_not_rewrite_earlier_user(context_entity):
    entity = context_entity
    saved = None
    model = {"id": "test-model", "info": {"params": {"system": "WORKSPACE"}}}

    async def request(method, path, headers, body=None):
        nonlocal saved
        if path == "/api/models":
            return {"data": [model]}
        if method == "GET":
            return {"chat": deepcopy(saved)}
        saved = deepcopy(body["chat"])
        return {"id": "saved"}

    entity._async_webui_request = request
    state = WebUIChat()
    entity.subentry.data[CONF_OPENWEBUI_ATTACH_HA_PROMPT] = True
    first = await entity._async_webui_prepare_turn(
        state,
        {
            "model": "test-model",
            "messages": [
                {"role": "system", "content": "HA_FIRST"},
                await entity._convert_content_to_chat_message(
                    conversation.UserContent(content="ONE"), "ha-conversation"
                ),
            ],
        },
        {},
        None,
    )
    first_id = first["user_message"]["id"]
    assert saved["history"]["messages"][first_id]["content"] == "HA_FIRST\n\nONE"
    entity.subentry.data[CONF_OPENWEBUI_ATTACH_HA_PROMPT] = False
    second = await entity._async_webui_prepare_turn(
        state,
        {
            "model": "test-model",
            "messages": [
                {"role": "system", "content": "HA_SECOND"},
                await entity._convert_content_to_chat_message(
                    conversation.UserContent(content="TWO"), "ha-conversation"
                ),
            ],
        },
        {},
        None,
    )
    assert second["user_message"]["content"] == "TWO"
    assert second["user_message"]["parentId"] == first["id"]
    assert saved["history"]["messages"][first["id"]]["childrenIds"] == [
        second["user_message"]["id"]
    ]
    assert saved["history"]["messages"][first_id]["content"] == "HA_FIRST\n\nONE"
    assert [m["content"] for m in saved["messages"] if m["role"] == "user"] == [
        "HA_FIRST\n\nONE",
        "TWO",
    ]
    assert "HA_SECOND" not in json.dumps(saved)


@pytest.mark.parametrize(
    "native,attach,pending,expected",
    [
        (True, False, False, False),
        (True, True, False, True),
        (True, True, True, False),
        (False, False, False, True),
    ],
)
async def test_prompt_preparation_only_when_used(
    context_entity, hass, native, attach, pending, expected
):
    entity = context_entity
    entity.subentry.data.update(
        {
            CONF_OPENWEBUI_NATIVE: native,
            CONF_OPENWEBUI_ATTACH_HA_PROMPT: attach,
            "prompt": "{{ invalid(",
        }
    )
    log = conversation.ChatLog(hass, "context-test")
    log.content.append(
        conversation.AssistantContent(agent_id=entity.entity_id, content="OK")
    )
    if pending:
        entity._webui_chats[log.conversation_id] = WebUIChat(
            pending={"call_id": "wait"}
        )
    entity._async_handle_chat_log = AsyncMock(return_value=False)
    with patch.object(type(log), "async_provide_llm_data", AsyncMock()) as prepare:
        await entity._async_handle_message(
            MagicMock(spec=conversation.ConversationInput), log
        )
    assert prepare.await_count == int(expected)


async def test_enabled_empty_prompt_retains_real_ha_default(context_entity, hass):
    entity = context_entity
    entity.subentry.data.update({CONF_OPENWEBUI_ATTACH_HA_PROMPT: True, "prompt": ""})
    log = conversation.ChatLog(hass, "default-test")
    log.async_add_user_content(conversation.UserContent(content="hello"))

    async def handle(chat_log, **kwargs):
        assert chat_log.content[0].content
        chat_log.content.append(
            conversation.AssistantContent(agent_id=entity.entity_id, content="OK")
        )
        return False

    entity._async_handle_chat_log = handle
    user_input = conversation.ConversationInput(
        text="hello",
        context=Context(),
        conversation_id=log.conversation_id,
        device_id=None,
        satellite_id=None,
        language="en",
        agent_id=entity.entity_id,
    )
    await entity._async_handle_message(user_input, log)
    assert log.content[0].role == "system" and log.content[0].content


@pytest.mark.parametrize("step", ["user", "reconfigure"])
async def test_checkbox_forms_and_translations(hass, mock_config_entry, step):
    flow = ConversationFlowHandler()
    flow.hass = hass
    mock_config_entry.runtime_data.models.list = AsyncMock(
        return_value=MagicMock(data=[])
    )
    flow._get_entry = MagicMock(return_value=mock_config_entry)
    flow._get_reconfigure_subentry = MagicMock(
        return_value=MagicMock(
            data={"model": "test-model", CONF_OPENWEBUI_ATTACH_HA_PROMPT: True}
        )
    )
    result = await getattr(flow, f"async_step_{step}")()
    schema = result["data_schema"].schema
    marker = next(key for key in schema if str(key) == CONF_OPENWEBUI_ATTACH_HA_PROMPT)
    assert schema[marker] is bool
    if step == "user":
        assert marker.default() is False
    else:
        assert marker.description["suggested_value"] is True
    folder = Path(__file__).parents[2] / "custom_components/local_openai/translations"
    for language in ("en", "de"):
        data = json.loads((folder / f"{language}.json").read_text())
        assert data["config_subentries"]["conversation"]["step"][step]["data"][
            CONF_OPENWEBUI_ATTACH_HA_PROMPT
        ]
