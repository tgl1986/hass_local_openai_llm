"""Native satellite pipeline regression; no physical device or live model."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from homeassistant.components import conversation, stt
from homeassistant.components.assist_satellite.entity import AssistSatelliteEntity
from homeassistant.components.assist_pipeline.pipeline import (
    async_get_pipeline,
    async_update_pipeline,
)
from homeassistant.exceptions import HomeAssistantError
from homeassistant.setup import async_setup_component

from test_audio import FixtureTTS, RECORDING


class FixtureSTT(stt.SpeechToTextEntity):
    _attr_name = "Satellite fixture STT"
    supported_languages = ["en"]
    supported_formats = [stt.AudioFormats.WAV]
    supported_codecs = [stt.AudioCodecs.PCM]
    supported_bit_rates = [stt.AudioBitRates.BITRATE_16]
    supported_sample_rates = [stt.AudioSampleRates.SAMPLERATE_16000]
    supported_channels = [stt.AudioChannels.CHANNEL_MONO]

    async def async_process_audio_stream(self, metadata, stream):
        return stt.SpeechResult("Wie ist das Wetter?", stt.SpeechResultState.SUCCESS)


class FixtureSatellite(AssistSatelliteEntity):
    def __init__(self):
        self.events = []
        self.states = []

    def async_get_configuration(self):
        raise NotImplementedError

    async def async_set_configuration(self, config):
        raise NotImplementedError

    def on_pipeline_event(self, event):
        self.events.append(event)
        self.states.append(self.state)


@pytest.mark.parametrize(
    "failure,streaming", [(False, False), (False, True), (True, False)]
)
async def test_native_satellite_response_and_completion(
    hass, hass_client, tmp_path, mock_conversation_entity, caplog, failure, streaming
):
    hass.config.config_dir = str(tmp_path)
    assert await async_setup_component(hass, "homeassistant", {})
    assert await async_setup_component(hass, "assist_pipeline", {})
    recognizer = FixtureSTT()
    await hass.data["stt"].async_add_entities([recognizer])
    voice = FixtureTTS(streaming)
    await hass.data["tts"].async_add_entities([voice])
    agent = mock_conversation_entity
    object.__setattr__(
        agent.subentry, "data", {"model": "fixture", "openwebui_native": True}
    )
    agent.model = "fixture"
    await hass.data["conversation"].async_add_entities([agent])
    agent._async_webui_connect = AsyncMock()
    agent._webui_socket = MagicMock()
    agent._webui_socket.get_sid.return_value = "fixture-session"
    converted_roles = []
    original_convert = agent._convert_content_to_chat_message

    async def convert(content, conversation_id):
        converted_roles.append(content.role)
        return await original_convert(content, conversation_id)

    agent._convert_content_to_chat_message = convert

    async def request(method, path, headers, body=None):
        if path == "/api/models":
            return {"data": [{"id": "fixture"}]}
        if path == "/api/v1/chats/new":
            return {"id": "fixture-chat"}
        if path == "/api/chat/completions":
            if failure:
                raise HomeAssistantError("Fixture backend failure")
            state = next(iter(agent._webui_chats.values()))
            for outer in RECORDING:
                state.events.put_nowait(outer["data"])
            return {"status": True}
        raise AssertionError(path)

    agent._async_webui_request = request
    pipeline = async_get_pipeline(hass)
    await async_update_pipeline(
        hass,
        pipeline,
        conversation_engine=agent.entity_id,
        conversation_language="en",
        prefer_local_intents=False,
        stt_engine=recognizer.entity_id,
        stt_language="en",
        tts_engine=voice.entity_id,
        tts_language="en",
    )
    satellite = FixtureSatellite()
    satellite.hass = hass
    satellite.entity_id = "assist_satellite.fixture"
    satellite.async_write_ha_state = MagicMock()
    satellite.platform = SimpleNamespace(
        config_entry=SimpleNamespace(
            async_create_background_task=lambda hass, coro, name: asyncio.create_task(
                coro, name=name
            )
        )
    )
    http = await hass_client()
    downloads = []
    original_event = satellite.on_pipeline_event

    async def download(url):
        response = await http.get(url)
        return response.status, await response.read()

    def on_event(event):
        original_event(event)
        if event.type == "run-start" and event.data.get("tts_output"):
            downloads.append(
                asyncio.create_task(download(event.data["tts_output"]["url"]))
            )

    satellite.on_pipeline_event = on_event

    async def audio():
        yield bytes(3200)

    try:
        await asyncio.wait_for(
            satellite.async_accept_pipeline_from_satellite(audio()), 5
        )
        types = [event.type for event in satellite.events]
        assert "stt-end" in types
        assert "processing" in satellite.states
        assert types.count("run-end") == 1
        assert satellite._pipeline_task is None
        assert "system" not in converted_roles
        assert "user" in converted_roles
        assert "Could not convert message to Completions API" not in caplog.text
        # HA returns conversation errors as spoken responses, not pipeline errors.
        assert "error" not in types
        assert "intent-end" in types and "tts-end" in types
        assert satellite.state == "responding"
        assert downloads
        for task in downloads:
            status, content = await asyncio.wait_for(task, 3)
            assert status == 200 and content
        spoken = "".join(voice.texts)
        assert ("Fixture backend failure" if failure else "Berlin") in spoken
        satellite.tts_response_finished()
        assert satellite.state == "idle"
    finally:
        for task in downloads:
            if not task.done():
                task.cancel()
        await asyncio.gather(*downloads, return_exceptions=True)


@pytest.mark.parametrize(
    "native,attach,prompt,roles",
    [
        (True, False, "", ["user"]),
        (True, False, "HA prompt", ["user"]),
        (True, True, "HA prompt", ["system", "user"]),
        (False, False, "HA prompt", ["system", "user"]),
    ],
)
async def test_system_prompt_handoff(
    hass, mock_conversation_entity, caplog, native, attach, prompt, roles
):
    agent = mock_conversation_entity
    object.__setattr__(
        agent.subentry,
        "data",
        {
            "model": "fixture",
            "openwebui_native": native,
            "openwebui_attach_ha_prompt": attach,
        },
    )
    log = conversation.ChatLog(hass, "handoff")
    log.content[0] = conversation.SystemContent(content=prompt)
    log.async_add_user_content(conversation.UserContent(content="Weather?"))
    original = agent._convert_content_to_chat_message
    agent._convert_content_to_chat_message = AsyncMock(wraps=original)
    args, _, _ = await agent._async_prepare_chat_log(
        log, list(log.content), None, None, None, parallel_tool_calls=False
    )
    assert [
        call.args[0].role
        for call in agent._convert_content_to_chat_message.call_args_list
    ] == roles
    assert [message["role"] for message in args["messages"]] == roles
    assert "Could not convert message to Completions API" not in caplog.text
