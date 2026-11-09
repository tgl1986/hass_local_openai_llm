"""Exercise installed HA websocket handler, pipeline and ChatLog unchanged.

Only the model/backend is replaced by a deterministic ConversationEntity.
Deadlines are shortened through the real request field, never monkeypatched.
"""

import asyncio
import json
import time

import pytest

from homeassistant.components import conversation
from homeassistant.components.assist_pipeline.pipeline import (
    async_get_pipeline,
    async_update_pipeline,
)
from homeassistant.components.conversation.const import DATA_COMPONENT
from homeassistant.setup import async_setup_component


class SlowStreamingAgent(conversation.ConversationEntity):
    _attr_name = "Timeout probe"
    _attr_supported_languages = ["en"]
    _attr_supports_streaming = True

    def __init__(self, mode):
        self.mode = mode
        self.cancelled = False
        self.completed = False

    @property
    def supported_languages(self):
        return ["en"]

    async def _async_handle_message(self, user_input, chat_log):
        async def stream():
            yield {"role": "assistant"}
            for _ in range(20):
                await asyncio.sleep(0.05)
                if self.mode == "text":
                    yield {"content": "working "}
                elif self.mode == "empty":
                    yield {"content": ""}
                elif self.mode == "thinking":
                    yield {"thinking_content": "thinking "}
            yield {"content": "finished"}

        try:
            async for _ in chat_log.async_add_delta_content_stream(
                self.entity_id, stream()
            ):
                pass
        except asyncio.CancelledError:
            self.cancelled = True
            raise
        self.completed = True
        return conversation.async_get_result_from_chat_log(user_input, chat_log)


@pytest.mark.parametrize(
    "mode,deadline,expected_timeout",
    [
        ("silent", 0.5, True),
        ("text", 0.5, True),
        ("empty", 0.5, True),
        ("thinking", 0.5, True),
        ("silent", 2, False),
        ("text", 2, False),
    ],
)
async def test_real_pipeline_streaming_deadline(
    hass, hass_ws_client, tmp_path, mode, deadline, expected_timeout
):
    hass.config.config_dir = str(tmp_path)
    assert await async_setup_component(hass, "homeassistant", {})
    assert await async_setup_component(hass, "assist_pipeline", {})
    agent = SlowStreamingAgent(mode)
    await hass.data[DATA_COMPONENT].async_add_entities([agent])
    pipeline = async_get_pipeline(hass)
    await async_update_pipeline(
        hass,
        pipeline,
        conversation_engine=agent.entity_id,
        conversation_language="en",
        prefer_local_intents=False,
    )
    client = await hass_ws_client(hass)
    started = time.monotonic()
    await client.send_json(
        {
            "id": 1,
            "type": "assist_pipeline/run",
            "start_stage": "intent",
            "end_stage": "intent",
            "input": {"text": "Run timeout probe"},
            "pipeline": pipeline.id,
            "timeout": deadline,
        }
    )
    result = await client.receive_json()
    assert result["success"], result
    events = []
    async with asyncio.timeout(5):
        while True:
            message = await client.receive_json()
            event = message["event"]
            events.append(event)
            if event["type"] == "error" or (
                not expected_timeout and event["type"] == "run-end"
            ):
                break
    elapsed = time.monotonic() - started
    errors = [e["data"] for e in events if e["type"] == "error"]
    deltas = [e for e in events if e["type"] == "intent-progress"]
    assert bool(errors) == expected_timeout, errors
    assert agent.cancelled == expected_timeout
    assert agent.completed != expected_timeout
    if expected_timeout:
        assert errors == [{"code": "timeout", "message": "Timeout running pipeline"}]
    if mode != "silent":
        assert len(deltas) >= 5
    print(
        "PROBE_RESULT "
        + json.dumps(
            {
                "mode": mode,
                "request_timeout_seconds": deadline,
                "elapsed_seconds": round(elapsed, 3),
                "progress_events": len(deltas),
                "cancelled": agent.cancelled,
                "completed": agent.completed,
                "errors": errors,
            }
        )
    )
