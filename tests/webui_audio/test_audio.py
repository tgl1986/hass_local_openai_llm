"""Replay actual WebUI events through product receive, real HA ChatLog, pipeline and TTS HTTP.

The TTS entity generates a valid WAV fixture, not real speech. No Core files are
modified. Model HTTP/socket transport is replaced by recorded WebUI events.
"""

import asyncio
import io
import json
import wave
from dataclasses import replace
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from homeassistant.components import conversation, tts
from homeassistant.components.conversation.const import (
    DATA_COMPONENT as CONVERSATION_COMPONENT,
)
from homeassistant.components.tts.const import DATA_COMPONENT as TTS_COMPONENT
from homeassistant.components.assist_pipeline.pipeline import (
    async_get_pipeline,
    async_update_pipeline,
)
from homeassistant.setup import async_setup_component
from custom_components.local_openai.entity import WebUIChat, GenerationMetadata
from custom_components.local_openai.conversation import LocalAiConversationEntity
from openai.types.chat import ChatCompletionChunk

RECORDING = json.loads((Path(__file__).parent / "events-1.json").read_text())
PREFIX = "Ich prüfe jetzt die Wetterdaten und warte auf die Antwort des Werkzeugs. " * 4
SHORT_PREFIX = "Ich prüfe das Wetter. "


def wav_bytes():
    output = io.BytesIO()
    with wave.open(output, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16000)
        wav.writeframes(b"\x00\x00" * 1600)
    return output.getvalue()


class FixtureTTS(tts.TextToSpeechEntity):
    _attr_name = "Audio replay TTS"
    _attr_supported_languages = ["en"]
    _attr_default_language = "en"
    _attr_supported_options = [
        "preferred_format",
        "preferred_sample_rate",
        "preferred_sample_channels",
        "preferred_sample_bytes",
    ]

    def __init__(self, streaming):
        self.streaming = streaming
        self.texts = []
        self.finished = False
        self.stream_called = False

    def async_supports_streaming_input(self):
        return self.streaming

    async def async_get_tts_audio(self, message, language, options):
        self.texts.append(message)
        self.finished = True
        return ("wav", wav_bytes())

    async def async_stream_tts_audio(self, request):
        self.stream_called = True

        async def audio():
            async for text in request.message_gen:
                self.texts.append(text)
            self.finished = True
            yield wav_bytes()

        return tts.TTSAudioResponse(extension="wav", data_gen=audio())


class ReplayAgent(conversation.ConversationEntity):
    _attr_name = "WebUI audio replay"
    _attr_supports_streaming = True

    def __init__(self, prefix):
        self.prefix = prefix
        self.at_tool_gap = asyncio.Event()
        self.continue_after_tool = asyncio.Event()
        self.completed = False
        self.chunks = []

    @property
    def supported_languages(self):
        return ["en"]

    async def _async_handle_message(self, user_input, chat_log):
        entry, subentry = MagicMock(), MagicMock()
        entry.data = {}
        subentry.data = {"model": "fixture", "openwebui_native": True}
        subentry.subentry_type = "conversation"
        native = LocalAiConversationEntity(entry, subentry)
        native.hass = self.hass
        native.entity_id = self.entity_id
        state = WebUIChat()

        async def feed():
            if self.prefix:
                state.events.put_nowait(
                    {
                        "type": "response:completion",
                        "data": {
                            "type": "response.output_text.delta",
                            "item_id": "pre-tool",
                            "delta": SHORT_PREFIX
                            if self.prefix in ("short", "no_followup")
                            else PREFIX,
                        },
                    }
                )
            for outer in RECORDING:
                event = outer["data"]
                if self.prefix == "no_followup":
                    if (
                        event.get("data", {}).get("type")
                        == "response.output_text.delta"
                    ):
                        continue
                    if "output" in event.get("data", {}):
                        event = {
                            **event,
                            "data": {
                                **event["data"],
                                "output": [
                                    item
                                    for item in event["data"]["output"]
                                    if item["type"] != "message"
                                ],
                            },
                        }
                state.events.put_nowait(event)
                data = event.get("data") or {}
                if (
                    data.get("type") == "response.output_item.done"
                    and data.get("item", {}).get("type") == "function_call"
                    and not self.at_tool_gap.is_set()
                ):
                    self.at_tool_gap.set()
                    await self.continue_after_tool.wait()

        async def product_stream():
            async for delta in native._async_webui_receive(state, GenerationMetadata()):
                self.chunks.append(delta)
                yield delta

        async def standard_stream():
            async def openai_events():
                yield ChatCompletionChunk(
                    id="fixture",
                    created=0,
                    model="fixture",
                    object="chat.completion.chunk",
                    choices=[
                        {
                            "index": 0,
                            "delta": {"role": "assistant", "content": SHORT_PREFIX},
                        }
                    ],
                )
                yield ChatCompletionChunk(
                    id="fixture",
                    created=0,
                    model="fixture",
                    object="chat.completion.chunk",
                    choices=[
                        {
                            "index": 0,
                            "finish_reason": "tool_calls",
                            "delta": {
                                "tool_calls": [
                                    {
                                        "index": 0,
                                        "id": "call_weather",
                                        "type": "function",
                                        "function": {
                                            "name": "get_weather",
                                            "arguments": '{"city":"Berlin"}',
                                        },
                                    }
                                ]
                            },
                        }
                    ],
                )

            async for delta in native._transform_stream(
                openai_events(), False, meta=GenerationMetadata()
            ):
                # The user's pre-WebUI installation explicitly used external=True.
                if "tool_calls" in delta:
                    delta["tool_calls"] = [
                        replace(tool, external=True) for tool in delta["tool_calls"]
                    ]
                self.chunks.append(delta)
                yield delta
            self.at_tool_gap.set()
            await self.continue_after_tool.wait()
            final = {"role": "assistant", "content": "In Berlin ist es sonnig. " * 4}
            self.chunks.append(final)
            yield final

        async def handle_chat(log, **kwargs):
            producer = (
                asyncio.create_task(feed()) if self.prefix != "standard" else None
            )
            try:
                stream = (
                    standard_stream() if self.prefix == "standard" else product_stream()
                )
                async for _ in log.async_add_delta_content_stream(
                    self.entity_id, stream
                ):
                    pass
            finally:
                if producer:
                    producer.cancel()
                    await asyncio.gather(producer, return_exceptions=True)
            self.completed = True
            return False

        native._async_handle_chat_log = handle_chat
        return await native._async_handle_message(user_input, chat_log)


@pytest.mark.parametrize(
    "streaming,prefix",
    [
        (False, False),
        (True, False),
        (True, True),
        (True, "short"),
        (True, "standard"),
        (False, "no_followup"),
        (True, "no_followup"),
    ],
)
async def test_tool_gap_keeps_turn_and_audio_open(
    hass, hass_ws_client, hass_client, tmp_path, streaming, prefix
):
    hass.config.config_dir = str(tmp_path)
    assert await async_setup_component(hass, "homeassistant", {})
    assert await async_setup_component(hass, "assist_pipeline", {})
    voice = FixtureTTS(streaming)
    await hass.data[TTS_COMPONENT].async_add_entities([voice])
    agent = ReplayAgent(prefix)
    await hass.data[CONVERSATION_COMPONENT].async_add_entities([agent])
    pipeline = async_get_pipeline(hass)
    await async_update_pipeline(
        hass,
        pipeline,
        conversation_engine=agent.entity_id,
        conversation_language="en",
        prefer_local_intents=False,
        tts_engine=voice.entity_id,
        tts_language="en",
    )
    ws = await hass_ws_client(hass)
    http = await hass_client()
    await ws.send_json(
        {
            "id": 1,
            "type": "assist_pipeline/run",
            "start_stage": "intent",
            "end_stage": "tts",
            "input": {"text": "Wie ist das Wetter?"},
            "pipeline": pipeline.id,
            "timeout": 10,
        }
    )
    accepted = await ws.receive_json()
    assert accepted["success"], accepted
    events = []
    audio_task = None

    async def read_audio(url):
        response = await http.get(url)
        return response.status, await response.read()

    async with asyncio.timeout(8):
        while True:
            event = (await ws.receive_json())["event"]
            events.append(event)
            if event["type"] == "run-start":
                audio_task = asyncio.create_task(
                    read_audio(event["data"]["tts_output"]["url"])
                )
                break
        await agent.at_tool_gap.wait()
        await asyncio.sleep(0.5)
        assert not agent.completed, "Product receive ended at ordinary tool completion"
        assert not voice.finished, "TTS input ended during tool gap"
        assert not audio_task.done(), "TTS HTTP ended before answer"
        streaming_during_gap = voice.stream_called
        assert streaming_during_gap == bool(streaming and prefix)
        agent.continue_after_tool.set()
        while True:
            event = (await ws.receive_json())["event"]
            events.append(event)
            assert event["type"] != "error", event
            if event["type"] == "run-end":
                break
        status, audio = await audio_task
    assert status == 200
    assert len(audio) > 0
    assert agent.completed and voice.finished
    spoken = "".join(voice.texts)
    if prefix == "no_followup":
        assert spoken == SHORT_PREFIX
        intent_end = next(event for event in events if event["type"] == "intent-end")
        assert (
            intent_end["data"]["intent_output"]["response"]["speech"]["plain"]["speech"]
            == SHORT_PREFIX
        )
    else:
        assert "Berlin" in spoken
    assert spoken == "".join(delta.get("content", "") for delta in agent.chunks)
    assert sum(event["type"] == "intent-end" for event in events) == 1
    assert sum(event["type"] == "run-end" for event in events) == 1
    started_streaming = any(
        (event.get("data") or {}).get("tts_start_streaming") for event in events
    )
    assert bool(started_streaming) == bool(streaming and prefix)
    print(
        "AUDIO_PROBE "
        + json.dumps(
            {
                "streaming": streaming,
                "prefix": prefix,
                "tts_stream_started": bool(started_streaming),
                "http_status": status,
                "tts_started_during_tool_gap": streaming_during_gap,
                "audio_bytes": len(audio),
                "tool_gap_turn_open": True,
                "tool_gap_audio_open": True,
                "text_complete": True,
                "intent_end_count": 1,
                "run_end_count": 1,
            }
        )
    )
