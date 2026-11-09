# ruff: noqa: F401, F811
# Imported native fixture is registered by pytest and requested by name.
"""Replay real WebUI wire recording through unchanged adapter and HA ChatLog."""

import asyncio
import json
from pathlib import Path
from custom_components.local_openai.entity import GenerationMetadata, WebUIChat
from test_webui import native as native, chatlog


async def test_actual_recorded_tool_reply_is_forwarded_before_done(native, hass):
    events = json.loads(
        (Path(__file__).parents[1] / "webui_audio" / "events-1.json").read_text()
    )
    state = WebUIChat()
    log = chatlog(hass)
    forwarded = []
    ready = asyncio.Event()

    def listener(_log, delta):
        if delta.get("content"):
            forwarded.append(delta["content"])
            ready.set()

    log.delta_listener = listener

    async def consume():
        async for _ in log.async_add_delta_content_stream(
            native.entity_id, native._async_webui_receive(state, GenerationMetadata())
        ):
            pass

    task = asyncio.create_task(consume())
    tool_seen = False
    immediate_count = 0
    try:
        for outer in events:
            event = outer["data"]
            data = event.get("data") or {}
            if data.get("item", {}).get("type") == "function_call_output":
                tool_seen = True
            ready.clear()
            before = len("".join(forwarded))
            state.events.put_nowait(event)
            if data.get("type") == "response.output_text.delta":
                assert tool_seen
                await asyncio.wait_for(ready.wait(), 1)
                assert "".join(forwarded)[before:] == data["delta"]
                assert not task.done()
                immediate_count += 1
        await asyncio.wait_for(task, 1)
        assert immediate_count > 0
        print(
            "RECORDED_STREAM "
            + json.dumps(
                {
                    "tool_result_before_text": tool_seen,
                    "text_deltas_forwarded_before_next_event": immediate_count,
                    "forwarded_characters": len("".join(forwarded)),
                    "waited_for_done": False,
                }
            )
        )
    finally:
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)
