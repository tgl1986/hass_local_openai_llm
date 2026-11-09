"""Focused diagnosis of actual initial role and tool_result -> assistant state."""

import asyncio
import json
from unittest.mock import MagicMock
from homeassistant.components import conversation
from homeassistant.helpers.chat_session import ChatSession
from custom_components.local_openai.entity import (
    LocalAiEntity,
    WebUIChat,
    GenerationMetadata,
)


class ObservedQueue(asyncio.Queue):
    def __init__(self):
        super().__init__()
        self.waiting = asyncio.Event()

    async def get(self):
        self.waiting.set()
        return await super().get()

    async def deliver(self, event):
        self.waiting.clear()
        self.put_nowait(event)
        await asyncio.wait_for(self.waiting.wait(), 2)


async def test_initial_role_is_not_completed_answer(hass, tmp_path):
    entry, subentry = MagicMock(), MagicMock()
    entry.data = {}
    subentry.data = {"model": "fixture", "openwebui_native": True}
    subentry.subentry_type = "conversation"
    entity = LocalAiEntity(entry, subentry)
    entity.hass = hass
    entity.entity_id = "conversation.fixture"
    state = WebUIChat()
    state.events = ObservedQueue()
    deltas = []
    completed = []
    states = []
    with conversation.async_get_chat_log(
        hass,
        ChatSession("initial-role-state"),
        chat_log_delta_listener=lambda log, d: deltas.append(d),
    ) as log:
        log.async_add_user_content(conversation.UserContent(content="fixture"))

        async def consume():
            async for content in log.async_add_delta_content_stream(
                entity.entity_id,
                entity._async_webui_receive(state, GenerationMetadata()),
            ):
                completed.append(content)

        task = asyncio.create_task(consume())

        def snapshot(phase):
            states.append(
                {
                    "phase": phase,
                    "turn_active": log.turn_active,
                    "consumer_done": task.done(),
                    "saved_roles": [c.role for c in log.content],
                    "saved_assistant_contents": [
                        c.content for c in log.content if c.role == "assistant"
                    ],
                    "completed_content_count": len(completed),
                    "last_delta": deltas[-1],
                }
            )

        try:
            await asyncio.wait_for(state.events.waiting.wait(), 2)
            snapshot("initial_role_received")
            assert deltas == [{"role": "assistant"}]
            assert log.turn_active and not task.done()
            assert [c.role for c in log.content] == ["system", "user"]
            assert not completed
            call = {
                "type": "function_call",
                "id": "call",
                "call_id": "call",
                "name": "weather",
                "arguments": "{}",
                "status": "completed",
            }
            result = {
                "type": "function_call_output",
                "id": "result",
                "call_id": "call",
                "output": "sunny",
            }
            for item in (call, result):
                await state.events.deliver(
                    {
                        "type": "response:completion",
                        "data": {"type": "response.output_item.done", "item": item},
                    }
                )
                snapshot(item["type"])
                assert log.turn_active and not task.done()
            assert log.content[-1].role == "tool_result"
            await state.events.deliver(
                {
                    "type": "response:completion",
                    "data": {
                        "type": "response.output_text.delta",
                        "item_id": "answer",
                        "delta": "Antwort.",
                    },
                }
            )
            snapshot("assistant_answer_delta")
            assert deltas[-2:] == [{"role": "assistant"}, {"content": "Antwort."}]
            assert log.turn_active and not task.done()
            state.events.put_nowait({"type": "chat:completion", "data": {"done": True}})
            await asyncio.wait_for(task, 2)
            snapshot("explicit_done_consumed")
            assert log.content[-1].content == "Antwort."
        finally:
            if not task.done():
                task.cancel()
            await asyncio.gather(task, return_exceptions=True)
    assert not log.turn_active
    states.append({"phase": "chatlog_context_exited", "turn_active": log.turn_active})
    (tmp_path / "empty-role-state.json").write_text(
        json.dumps(states, indent=2, default=str)
    )
    print("EMPTY_ROLE_STATE " + json.dumps(states, default=str), flush=True)
