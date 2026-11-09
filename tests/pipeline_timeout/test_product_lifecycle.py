"""Exercise actual product entity registration, removal and config forms."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from homeassistant.components import websocket_api
from homeassistant.components.assist_pipeline.pipeline import (
    async_get_pipeline,
    async_update_pipeline,
)
from homeassistant.components.conversation.const import DATA_COMPONENT
from homeassistant.setup import async_setup_component

from custom_components.local_openai.config_flow import ConversationFlowHandler
from custom_components.local_openai.const import (
    CONF_PIPELINE_TIMEOUT,
    DEFAULT_PIPELINE_TIMEOUT,
)
from custom_components.local_openai.conversation import LocalAiConversationEntity
from custom_components.local_openai.pipeline_timeout import COMMAND, DATA_KEY

from .probe_agent import SlowStreamingAgent
from .test_pipeline_start import run_request


class SlowLocalAgent(LocalAiConversationEntity):
    """Use the product lifecycle; substitute only model response generation."""

    mode = "silent"
    cancelled = False
    completed = False
    _async_handle_message = SlowStreamingAgent._async_handle_message


@pytest.mark.parametrize(
    "setting,expected,preload", [(None, 1800, True), (2, 2, True), (None, 1800, False)]
)
async def test_real_entity_registers_and_unloads(
    hass, hass_ws_client, tmp_path, setting, expected, preload
):
    hass.config.config_dir = str(tmp_path)
    assert await async_setup_component(hass, "homeassistant", {})
    assert await async_setup_component(
        hass, "assist_pipeline" if preload else "conversation", {}
    )
    client = await hass_ws_client(hass)
    original = hass.data[websocket_api.DOMAIN].get(COMMAND)
    if not preload:
        assert original is None
    entry = MagicMock()
    entry.data = {}
    subentry = MagicMock()
    subentry.data = {"model": "test-model"}
    subentry.subentry_type = "conversation"
    subentry.subentry_id = "timeout-product-test"
    subentry.title = "Timeout product test"
    if setting is not None:
        subentry.data[CONF_PIPELINE_TIMEOUT] = setting
    entity = SlowLocalAgent(entry, subentry)
    await hass.data[DATA_COMPONENT].async_add_entities([entity])
    if original is None:
        state = hass.data[DATA_KEY]
        original = (state.original, state.schema)
    assert hass.data[websocket_api.DOMAIN][COMMAND][0] is not original[0]
    pipeline = async_get_pipeline(hass)
    await async_update_pipeline(
        hass,
        pipeline,
        conversation_engine=entity.entity_id,
        conversation_language="en",
        prefer_local_intents=False,
    )
    effective, _ = await run_request(client, pipeline.id, 0.3)
    assert effective == expected
    assert entity.completed and not entity.cancelled
    await entity.async_remove()
    assert hass.data[websocket_api.DOMAIN][COMMAND] == original
    assert DATA_KEY not in hass.data
    print(
        f"PRODUCT_LIFECYCLE setting={setting} effective={effective} preload={preload} completed=True restored=True"
    )


@pytest.mark.parametrize("step", ["user", "reconfigure"])
async def test_timeout_field_in_real_forms(hass, step):
    entry = MagicMock()
    entry.data = {}
    entry.runtime_data.models.list = AsyncMock(return_value=MagicMock(data=[]))
    subentry = MagicMock()
    subentry.data = {"model": "test-model", CONF_PIPELINE_TIMEOUT: 2400}
    handler = ConversationFlowHandler()
    handler.hass = hass
    handler._get_entry = MagicMock(return_value=entry)
    handler._get_reconfigure_subentry = MagicMock(return_value=subentry)
    handler.get_llm_apis = MagicMock(return_value=[])
    result = await getattr(handler, f"async_step_{step}")()
    assert result["type"] == "form"
    schema = result["data_schema"].schema
    marker = next(key for key in schema if str(key) == CONF_PIPELINE_TIMEOUT)
    assert schema[marker](2400) == 2400
    if step == "user":
        assert marker.default() == DEFAULT_PIPELINE_TIMEOUT == 1800
    else:
        assert marker.description["suggested_value"] == 2400
