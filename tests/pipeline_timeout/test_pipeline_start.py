"""Actual HA websocket starts through the integration-side prototype."""

import asyncio
import json

import pytest

from homeassistant.components import websocket_api
from homeassistant.components.assist_pipeline.pipeline import (
    async_get_pipeline,
    async_update_pipeline,
)
from homeassistant.components.conversation.const import DATA_COMPONENT
from homeassistant.setup import async_setup_component

from custom_components.local_openai.pipeline_timeout import (
    COMMAND,
    DATA_KEY,
    async_register_agent_timeout,
)
from .probe_agent import SlowStreamingAgent


@pytest.fixture
async def setup_probe(hass, hass_ws_client, tmp_path):
    hass.config.config_dir = str(tmp_path)
    assert await async_setup_component(hass, "homeassistant", {})
    assert await async_setup_component(hass, "assist_pipeline", {})
    agent = SlowStreamingAgent("text")
    await hass.data[DATA_COMPONENT].async_add_entities([agent])
    pipeline = async_get_pipeline(hass)
    await async_update_pipeline(
        hass,
        pipeline,
        conversation_engine=agent.entity_id,
        conversation_language="en",
        prefer_local_intents=False,
    )
    # Open before installing the hook: prove existing connections see it too.
    client = await hass_ws_client(hass)
    return agent, pipeline, client


async def run_request(client, pipeline_id, deadline, expected_timeout=False):
    msg = {
        "type": COMMAND,
        "start_stage": "intent",
        "end_stage": "intent",
        "input": {"text": "Test central start timeout"},
    }
    if pipeline_id is not None:
        msg["pipeline"] = pipeline_id
    if deadline is not None:
        msg["timeout"] = deadline
    await client.send_json_auto_id(msg)
    result = await client.receive_json()
    assert result["success"], result
    events = []
    async with asyncio.timeout(5):
        while True:
            event = (await client.receive_json())["event"]
            events.append(event)
            if event["type"] == "error" or (
                event["type"] == "run-end" and not expected_timeout
            ):
                break
    errors = [e["data"] for e in events if e["type"] == "error"]
    assert bool(errors) == expected_timeout, errors
    if expected_timeout:
        assert errors == [{"code": "timeout", "message": "Timeout running pipeline"}]
    effective = next(
        e["data"]["runner_data"]["timeout"] for e in events if e["type"] == "run-start"
    )
    return effective, events


@pytest.mark.parametrize(
    "selection,configured,incoming,expected",
    [
        ("id", 2, 0.3, 2),
        ("preferred", 2, 0.3, 2),
        ("entity_alias", 2, 0.3, 2),
        ("id", 2, 4, 4),
        ("id", 1800, None, 1800),
    ],
)
async def test_target_start(
    hass, setup_probe, selection, configured, incoming, expected
):
    agent, pipeline, client = setup_probe
    original = hass.data[websocket_api.DOMAIN][COMMAND]
    remove = await async_register_agent_timeout(
        hass, lambda: (agent.entity_id, configured)
    )
    try:
        assert hass.data[websocket_api.DOMAIN][COMMAND][1] is original[1]
        chosen = {
            "id": pipeline.id,
            "preferred": None,
            "entity_alias": agent.entity_id,
        }[selection]
        effective, events = await run_request(client, chosen, incoming)
        assert effective == expected
        assert agent.completed and not agent.cancelled
        print(
            "START_PROBE "
            + json.dumps(
                {
                    "selection": selection,
                    "incoming": incoming,
                    "configured": configured,
                    "effective": effective,
                    "completed": True,
                    "progress_events": sum(
                        e["type"] == "intent-progress" for e in events
                    ),
                }
            )
        )
    finally:
        remove()
    assert hass.data[websocket_api.DOMAIN][COMMAND] == original


async def test_foreign_agent_unchanged(hass, setup_probe):
    agent, pipeline, client = setup_probe
    remove = await async_register_agent_timeout(
        hass, lambda: ("conversation.another_agent", 2)
    )
    try:
        effective, _ = await run_request(
            client, pipeline.id, 0.3, expected_timeout=True
        )
        assert effective == 0.3 and agent.cancelled
    finally:
        remove()


async def test_silent_prefill_target(hass, setup_probe):
    agent, pipeline, client = setup_probe
    agent.mode = "silent"
    remove = await async_register_agent_timeout(hass, lambda: (agent.entity_id, 2))
    try:
        effective, events = await run_request(client, pipeline.id, 0.3)
        assert effective == 2 and agent.completed and not agent.cancelled
        # Initial role, then final answer: no periodic keepalive content.
        assert sum(e["type"] == "intent-progress" for e in events) == 2
        print(
            "START_PROBE silent_prefill: timeout 0.3 -> 2; completed without periodic deltas"
        )
    finally:
        remove()


async def test_multiple_registrations_and_unload(hass, setup_probe):
    agent, pipeline, client = setup_probe
    original = hass.data[websocket_api.DOMAIN][COMMAND]
    remove_first = await async_register_agent_timeout(
        hass, lambda: ("conversation.other", 5)
    )
    handler = hass.data[websocket_api.DOMAIN][COMMAND][0]
    remove_second = await async_register_agent_timeout(
        hass, lambda: (agent.entity_id, 2)
    )
    assert hass.data[websocket_api.DOMAIN][COMMAND][0] is handler
    remove_first()
    assert hass.data[websocket_api.DOMAIN][COMMAND][0] is handler
    effective, _ = await run_request(client, pipeline.id, 0.3)
    assert effective == 2
    remove_second()
    remove_second()  # idempotent callback
    assert hass.data[websocket_api.DOMAIN][COMMAND] == original
    assert DATA_KEY not in hass.data
    effective, _ = await run_request(client, pipeline.id, 0.3, expected_timeout=True)
    assert effective == 0.3


async def test_invalid_pipeline_retains_original_error(hass, setup_probe):
    agent, _, client = setup_probe
    remove = await async_register_agent_timeout(hass, lambda: (agent.entity_id, 2))
    try:
        await client.send_json_auto_id(
            {
                "type": COMMAND,
                "start_stage": "intent",
                "end_stage": "intent",
                "input": {"text": "test"},
                "pipeline": "missing",
            }
        )
        response = await client.receive_json()
        assert not response["success"]
        assert response["error"]["code"] == "pipeline-not-found"
    finally:
        remove()


async def test_unsubscribe_still_cancels(hass, setup_probe):
    agent, pipeline, client = setup_probe
    remove = await async_register_agent_timeout(hass, lambda: (agent.entity_id, 2))
    try:
        await client.send_json(
            {
                "id": 1,
                "type": COMMAND,
                "start_stage": "intent",
                "end_stage": "intent",
                "input": {"text": "test"},
                "pipeline": pipeline.id,
                "timeout": 0.3,
            }
        )
        assert (await client.receive_json())["success"]
        while (await client.receive_json())["event"]["type"] != "intent-progress":
            pass
        await client.send_json(
            {"id": 2, "type": "unsubscribe_events", "subscription": 1}
        )
        async with asyncio.timeout(3):
            while not agent.cancelled:
                await asyncio.sleep(0.01)
        assert not agent.completed
    finally:
        remove()


async def test_later_foreign_handler_not_overwritten(hass, setup_probe):
    agent, _, _ = setup_probe
    original = hass.data[websocket_api.DOMAIN][COMMAND]
    remove = await async_register_agent_timeout(hass, lambda: (agent.entity_id, 2))

    def other_handler(*args):
        pass

    websocket_api.async_register_command(hass, COMMAND, other_handler, original[1])
    remove()
    assert hass.data[websocket_api.DOMAIN][COMMAND][0] is other_handler
    websocket_api.async_register_command(hass, COMMAND, *original)
