"""Base entity for Open Router."""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import time
import uuid
from builtins import anext
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import timedelta
from typing import TYPE_CHECKING, Any, Literal

import aiohttp
import demoji
import openai
import socketio
from homeassistant.components import conversation
from homeassistant.const import CONF_MODEL
from homeassistant.core import callback
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers import device_registry as dr
from homeassistant.helpers import llm, template
from homeassistant.helpers.aiohttp_client import async_get_clientsession
from homeassistant.helpers.entity import Entity
from homeassistant.helpers.event import async_track_time_interval
from homeassistant.util import dt as dt_util
from homeassistant.util import slugify
from openai.types.chat import (
    ChatCompletionAssistantMessageParam,
    ChatCompletionContentPartImageParam,
    ChatCompletionContentPartTextParam,
    ChatCompletionFunctionToolParam,
    ChatCompletionMessageFunctionToolCallParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionToolMessageParam,
    ChatCompletionUserMessageParam,
)
from openai.types.chat.chat_completion_message_function_tool_call_param import Function
from openai.types.shared_params import FunctionDefinition, ResponseFormatJSONSchema
from yarl import URL

try:
    from probatio import to_openapi as convert
except ImportError:
    from voluptuous_openapi import convert

from .const import (
    CONF_CHAT_TEMPLATE_KWARGS,
    CONF_CHAT_TEMPLATE_OPTS,
    CONF_CONTENT_INJECTION_METHOD,
    CONF_CONTENT_INJECTION_METHOD_ASSISTANT,
    CONF_CONTENT_INJECTION_METHOD_TOOL,
    CONF_CONTENT_INJECTION_METHOD_USER,
    CONF_CUSTOM_HEADERS,
    CONF_MAX_MESSAGE_HISTORY,
    CONF_OPENWEBUI_ATTACH_HA_PROMPT,
    CONF_OPENWEBUI_NATIVE,
    CONF_PASS_SESSION_ID,
    CONF_REQUEST_BODY_OPTS,
    CONF_REQUEST_BODY_PARAMETERS,
    CONF_SERVER_HEADERS,
    CONF_SERVER_OPTIONS,
    CONF_STRIP_EMOJIS,
    CONF_TEMPERATURE,
    CONF_WEAVIATE_API_KEY,
    CONF_WEAVIATE_CLASS_NAME,
    CONF_WEAVIATE_DEFAULT_CLASS_NAME,
    CONF_WEAVIATE_DEFAULT_HYBRID_SEARCH_ALPHA,
    CONF_WEAVIATE_DEFAULT_MAX_RESULTS,
    CONF_WEAVIATE_DEFAULT_THRESHOLD,
    CONF_WEAVIATE_HOST,
    CONF_WEAVIATE_HYBRID_SEARCH_ALPHA,
    CONF_WEAVIATE_MAX_RESULTS,
    CONF_WEAVIATE_OPTIONS,
    CONF_WEAVIATE_THRESHOLD,
    DOMAIN,
    TOOL_NAME_STOP_TALKING,
)
from .weaviate import WeaviateClient

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Callable
    from datetime import datetime
    from pathlib import Path

    import voluptuous as vol
    from homeassistant.config_entries import ConfigSubentry
    from openai._streaming import AsyncStream
    from openai.types.chat import (
        ChatCompletionChunk,
        ChatCompletionMessageParam,
        ChatCompletionMessageToolCall,
    )
    from openai.types.shared_params.response_format_json_schema import JSONSchema

    from . import LocalAiConfigEntry


@dataclass
class GenerationMetadata:
    """Structured metadata collected during LLM generation."""

    request_start: float = 0
    time_to_first_token: float = 0
    tool_names: list[str] | None = None
    timings: dict | None = None
    usage: dict | None = None
    finish_reason: str | None = None


@dataclass
class AgentContext:
    """Per-iteration context for the agent loop."""

    iteration: int = 0
    conversation_id: str | None = None
    extra_body: dict = field(default_factory=dict)


@dataclass
class WebUIChat:
    """Identifiers and receive position for one native saved chat."""

    chat_id: str = ""
    assistant_id: str = ""
    model_id: str = ""
    pending: dict | None = None
    answers: dict[str, str] = field(default_factory=dict)
    text: dict[tuple[str, int, str], str] = field(default_factory=dict)
    tool_calls: dict[str, str] = field(default_factory=dict)
    tool_results: set[str] = field(default_factory=set)
    events: asyncio.Queue = field(default_factory=asyncio.Queue)
    active: bool = False
    headers: dict[str, str] = field(default_factory=dict)


_LOGGER = logging.getLogger(__name__)

# Max number of back and forth with the LLM to generate a response
MAX_TOOL_ITERATIONS = 10

# Check if HA supports thinking_content (2026.4+)
_SUPPORTS_THINKING = "thinking_content" in getattr(
    conversation.AssistantContent,
    "__dataclass_fields__",
    {},
)


def _remove_unsupported_keys_from_tool_schema(schema: dict[str, Any]) -> None:
    """Remove keys not supported in the tool schema."""
    for key in ("allOf", "anyOf", "oneOf"):
        schema.pop(key, None)


def _adjust_schema(schema: dict[str, Any]) -> None:
    """Adjust the schema to be compatible with structured output requirements."""
    if schema["type"] == "object":
        if "properties" not in schema:
            return

        if "required" not in schema:
            schema["required"] = []

        # Ensure all properties are required
        for prop, prop_info in schema["properties"].items():
            _adjust_schema(prop_info)
            if prop not in schema["required"]:
                prop_info["type"] = [prop_info["type"], "null"]
                schema["required"].append(prop)

    elif schema["type"] == "array":
        if "items" not in schema:
            return

        _adjust_schema(schema["items"])


def _format_structured_output(
    name: str,
    schema: vol.Schema,
    llm_api: llm.APIInstance | None,
) -> JSONSchema:
    """Format the schema to be compatible with OpenAI API."""
    result: JSONSchema = {
        # The API requires json_schema.name to match ^[a-zA-Z0-9_-]+$ and be
        # <= 64 chars; task names may contain spaces/punctuation. Mirror HA
        # core's openai_conversation which slugifies this value. Use a fallback
        # in case the name is empty or None (HA's slugify() returns '' for
        # both, but 'unknown' for other unsluggable input).
        "name": slugify(name)[:64] or "task",
        "strict": True,
    }
    result_schema = convert(
        schema,
        custom_serializer=(
            llm_api.custom_serializer if llm_api else llm.selector_serializer
        ),
    )

    _adjust_schema(result_schema)

    result["schema"] = result_schema
    return result


def _format_tool(
    tool: llm.Tool,
    custom_serializer: Callable[[Any], Any] | None,
) -> ChatCompletionFunctionToolParam:
    """Format tool specification."""
    parameters = convert(tool.parameters, custom_serializer=custom_serializer)
    _remove_unsupported_keys_from_tool_schema(parameters)

    tool_spec = FunctionDefinition(
        name=tool.name,
        parameters=parameters,
    )
    tool_spec["description"] = (
        tool.description
        if tool.description is not None and tool.description.strip()
        else "A callable function"
    )
    return ChatCompletionFunctionToolParam(type="function", function=tool_spec)


def b64_file(file_path: Path) -> str:
    """Retrieve the base64 encoded file contents."""
    return base64.b64encode(file_path.read_bytes()).decode("utf-8")


def _make_uuid(identifier: str) -> str:
    return str(uuid.uuid5(namespace=uuid.NAMESPACE_OID, name=identifier))


class LocalAiEntity(Entity):
    """Base entity for Open Router."""

    _attr_has_entity_name = True

    def __init__(self, entry: LocalAiConfigEntry, subentry: ConfigSubentry) -> None:
        """Initialize the entity."""
        self.entry = entry
        self.subentry = subentry
        self.model = subentry.data[CONF_MODEL]
        self._attr_unique_id = subentry.subentry_id
        self._webui_chats: dict[str, WebUIChat] = {}
        self._webui_socket: socketio.AsyncClient | None = None
        self._webui_connect_lock = asyncio.Lock()
        self._webui_heartbeat_unsub: Callable[[], None] | None = None
        self._attr_device_info = dr.DeviceInfo(
            identifiers={(DOMAIN, subentry.subentry_id)},
            name=subentry.title,
            entry_type=dr.DeviceEntryType.SERVICE,
        )

    async def _async_get_model(
        self, chat_log: conversation.ChatLog | None = None
    ) -> str:
        """Return the model to use."""
        return self.model

    @property
    def options(self) -> dict:
        """Return subentry data options."""
        return self.subentry.data

    @property
    def server_options(self) -> dict:
        """Return server options from entry data."""
        return self.entry.data.get(CONF_SERVER_OPTIONS, {})

    @property
    def weaviate_server_opts(self) -> dict:
        """Return Weaviate server options from entry data."""
        return self.entry.data.get(CONF_WEAVIATE_OPTIONS, {})

    @property
    def _uses_webui(self) -> bool:
        return self.subentry.subentry_type == "conversation" and self.options.get(
            CONF_OPENWEBUI_NATIVE, False
        )

    def _webui_has_pending(self, conversation_id: str) -> bool:
        state = self._webui_chats.get(conversation_id)
        return bool(state and state.pending)

    async def async_added_to_hass(self) -> None:
        """Follow HA's chat lifecycle instead of introducing another timeout."""
        await super().async_added_to_hass()
        if self._uses_webui:
            from homeassistant.components.conversation.chat_log import (  # noqa: PLC0415
                async_subscribe_chat_logs,
            )

            self.async_on_remove(
                async_subscribe_chat_logs(self.hass, self._webui_chat_event)
            )

    @callback
    def _webui_chat_event(
        self, conversation_id: str, event_type: str, _data: dict
    ) -> None:
        if event_type != "deleted":
            return
        state = self._webui_chats.pop(conversation_id, None)
        if state and state.active:
            # Wake the waiter; the existing cancellation path forwards Stop.
            state.events.put_nowait(asyncio.CancelledError())

    def _webui_base_url(self) -> URL:
        """Keep reverse-proxy prefixes when deriving the native API root."""
        url = URL(str(self.entry.runtime_data.base_url))
        path = url.path.rstrip("/")
        for suffix in ("/api/v1", "/api", "/v1"):
            if path.endswith(suffix):
                path = path[: -len(suffix)]
                break
        return url.with_path(path)

    async def _async_webui_request(
        self, method: str, path: str, headers: dict, body: dict | None = None
    ) -> dict:
        url = str(self._webui_base_url()).rstrip("/") + path
        request_headers = {
            "Authorization": f"Bearer {self.entry.runtime_data.api_key}",
            **headers,
        }
        async with async_get_clientsession(self.hass).request(
            method,
            url,
            headers=request_headers,
            json=body,
            timeout=aiohttp.ClientTimeout(total=None),
        ) as response:
            response.raise_for_status()
            return await response.json()

    async def _async_webui_connect(self, headers: dict) -> None:
        async with self._webui_connect_lock:
            if self._webui_socket is None:
                self._webui_socket = socketio.AsyncClient(
                    reconnection=False,
                    request_timeout=None,
                    http_session=async_get_clientsession(self.hass),
                    handle_sigint=False,
                )

                async def on_event(event: dict) -> None:
                    for state in self._webui_chats.values():
                        if (
                            event.get("chat_id") == state.chat_id
                            and event.get("message_id") == state.assistant_id
                        ):
                            state.events.put_nowait(event["data"])

                async def on_disconnect(*_args: Any) -> None:
                    self._webui_stop_heartbeat()
                    for state in self._webui_chats.values():
                        if state.active:
                            state.events.put_nowait(
                                HomeAssistantError(
                                    "OpenWebUI connection lost; the server task may still be running."
                                )
                            )

                self._webui_socket.on("events", on_event)
                self._webui_socket.on("disconnect", on_disconnect)

            if not self._webui_socket.connected:
                config = await self._async_webui_request("GET", "/api/config", headers)
                heartbeat_interval = (config.get("features") or {}).get(
                    "websocket_heartbeat_interval"
                )
                if heartbeat_interval is None:
                    heartbeat_interval = 30
                url = self._webui_base_url()
                await self._webui_socket.connect(
                    str(url.origin()),
                    socketio_path=url.path.strip("/") + "/ws/socket.io",
                    auth={"token": self.entry.runtime_data.api_key},
                    headers=headers,
                    transports=["websocket"],
                    wait_timeout=None,
                )
                self._webui_stop_heartbeat()
                self._webui_heartbeat_unsub = async_track_time_interval(
                    self.hass,
                    self._async_webui_heartbeat,
                    timedelta(seconds=heartbeat_interval),
                )

    async def _async_webui_heartbeat(self, _now: datetime) -> None:
        """Maintain WebUI's application session independently of model activity."""
        if self._webui_socket is not None and self._webui_socket.connected:
            await self._webui_socket.emit("heartbeat", {})

    @callback
    def _webui_stop_heartbeat(self) -> None:
        if self._webui_heartbeat_unsub is not None:
            self._webui_heartbeat_unsub()
            self._webui_heartbeat_unsub = None

    async def async_will_remove_from_hass(self) -> None:
        """Release this entity's native transport, preserving its saved chats."""
        self._webui_stop_heartbeat()
        try:
            for state in tuple(self._webui_chats.values()):
                if state.active and state.chat_id:
                    try:
                        await self._async_webui_stop(state)
                    except Exception:
                        _LOGGER.exception(
                            "Could not stop OpenWebUI task while unloading"
                        )
                    finally:
                        state.events.put_nowait(HomeAssistantError("Agent unloaded."))
        finally:
            try:
                if self._webui_socket is not None:
                    await self._webui_socket.disconnect()
            finally:
                self._webui_chats.clear()
                await super().async_will_remove_from_hass()

    async def _async_webui_stop(self, state: WebUIChat) -> None:
        result = await self._async_webui_request(
            "POST", f"/api/tasks/chat/{state.chat_id}/stop", state.headers
        )
        if not result.get("status"):
            msg = "OpenWebUI did not confirm stopping the chat task."
            raise HomeAssistantError(msg)

    @staticmethod
    def _webui_question(state: WebUIChat) -> str:
        call = state.pending
        if call["name"] != "ask_user":
            return (
                f"{call['name']}: {call.get('arguments', '')}. Ausführen? Ja oder Nein?"
            )
        args = json.loads(call["arguments"])
        question = next(q for q in args["questions"] if q["id"] not in state.answers)
        choices = "; ".join(
            f"{index}: {option['label']} — {option['description']}"
            for index, option in enumerate(question["options"], 1)
        )
        return f"{question['question']} {choices}"

    @staticmethod
    def _webui_answer(state: WebUIChat, text: str) -> dict | None:
        """Map explicit UI input to the native resolver, without another LLM."""
        call = state.pending
        choice = text.strip().rstrip(".?!").casefold()
        if call["name"] != "ask_user":
            if choice in ("ja", "yes", "approve", "freigeben", "bestätigen"):
                return {"call_id": call["call_id"], "action": "approve"}
            if choice in ("nein", "no", "reject", "ablehnen"):
                return {"call_id": call["call_id"], "action": "reject"}
            return None
        args = json.loads(call["arguments"])
        question = next(q for q in args["questions"] if q["id"] not in state.answers)
        answer = (
            text.strip()
            if question.get("allow_other", args.get("allow_other", True))
            else None
        )
        for index, option in enumerate(question["options"], 1):
            if choice in (str(index), option["label"].casefold()):
                answer = option["label"]
                break
        if not answer:
            return None
        state.answers[question["id"]] = answer
        if any(q["id"] not in state.answers for q in args["questions"]):
            return None
        return {
            "call_id": call["call_id"],
            "action": "answer",
            "answers": dict(state.answers),
        }

    @staticmethod
    def _webui_workspace_options(model: dict) -> dict:
        """Copy configured selection; capability/permission enforcement stays upstream."""
        meta = model.get("info", {}).get("meta") or {}
        # These are selection identifiers from WebUI, not a list of tools or
        # capabilities maintained by this integration.
        result = {
            "tool_ids": list(meta.get("toolIds") or []),
            "features": dict.fromkeys(meta.get("defaultFeatureIds") or [], True),
        }
        for source, target in (
            ("terminalId", "terminal_id"),
            ("skillIds", "skill_ids"),
            ("defaultFilterIds", "filter_ids"),
        ):
            if source in meta:
                result[target] = deepcopy(meta[source])
        return result

    async def _async_webui_prepare_turn(
        self, state: WebUIChat, model_args: dict, extra: dict, injected: dict | None
    ) -> dict:
        models = await self._async_webui_request("GET", "/api/models", state.headers)
        model = next(
            (m for m in models["data"] if m["id"] == model_args["model"]), None
        )
        if model is None:
            msg = "Configured OpenWebUI Workspace model was not found."
            raise HomeAssistantError(msg)
        # Preserve ordinary request options. Only native protocol and Workspace
        # selection fields are supplied by this path instead of by the SDK.
        body = deepcopy({k: v for k, v in model_args.items() if k != "extra_headers"})
        body.update(deepcopy(extra))
        body.pop("tools", None)
        body.pop("tool_choice", None)
        for key in ("terminal_id", "skill_ids", "filter_ids", "tool_servers"):
            body.pop(key, None)
        workspace = self._webui_workspace_options(model)
        body.update(workspace)
        params = {
            **(model.get("info", {}).get("params") or {}),
            **(body.get("params") or {}),
        }
        # The native resolver rebuilds requests from saved params. Keep existing
        # generation options there as well, excluding native routing/context.
        protocol_fields = {
            "model",
            "messages",
            "params",
            "stream",
            "metadata",
            "tool_ids",
            "features",
            "terminal_id",
            "skill_ids",
            "filter_ids",
            "files",
            "variables",
            "chat_variables",
            "background_tasks",
            "session_id",
            "chat_id",
            "id",
            "user_message",
            "parent_id",
            "assistant_message_id",
        }
        generation = {
            key: deepcopy(value)
            for key, value in body.items()
            if key not in protocol_fields
        }
        params.update(generation)
        if params.get("custom_params"):
            params["custom_params"] = {
                key: generation.get(key, value)
                for key, value in params["custom_params"].items()
            }
        params["function_calling"] = "native"
        system = (
            next(
                (m["content"] for m in body["messages"] if m["role"] == "system"), None
            )
            if self.options.get(CONF_OPENWEBUI_ATTACH_HA_PROMPT, False)
            else None
        )
        # Let WebUI resolve its Workspace system prompt from the model ID.
        params.pop("system", None)
        body.pop("system", None)
        body["messages"] = [m for m in body["messages"] if m["role"] != "system"]
        body["params"] = params
        if state.chat_id:
            saved = await self._async_webui_request(
                "GET", f"/api/v1/chats/{state.chat_id}", state.headers
            )
            chat = saved["chat"]
        else:
            chat = {
                "title": "Home Assistant",
                "models": [model["id"]],
                "history": {"messages": {}, "currentId": None},
            }
        history = chat["history"]["messages"]
        parent_id = state.assistant_id or None
        user_id, assistant_id = str(uuid.uuid4()), str(uuid.uuid4())
        user = deepcopy(body["messages"][-1])
        if system:
            content = user["content"]
            user["content"] = (
                f"{system}\n\n{content}"
                if isinstance(content, str)
                else [{"type": "text", "text": system}, *content]
            )
            body["messages"][-1]["content"] = deepcopy(user["content"])
        # Saved WebUI messages are rendered as strings, unlike Completions parts.
        content = user["content"]
        if isinstance(content, list) and all(
            part.get("type") == "text" for part in content
        ):
            user["content"] = "\n\n".join(part["text"] for part in content)
        user.update(
            id=user_id,
            parentId=parent_id,
            childrenIds=[assistant_id],
            timestamp=int(time.time()),
            models=[model["id"]],
        )
        additions = []
        if injected is not None:
            context_id = str(uuid.uuid4())
            context_message = deepcopy(injected)
            context_message.update(
                id=context_id,
                parentId=parent_id,
                childrenIds=[user_id],
                timestamp=int(time.time()),
            )
            additions.append(context_message)
            user["parentId"] = context_id
        assistant = {
            "id": assistant_id,
            "parentId": user_id,
            "childrenIds": [],
            "role": "assistant",
            "content": "",
            "done": False,
            "model": model["id"],
            "modelName": model.get("name", model["id"]),
            "modelIdx": 0,
            "timestamp": int(time.time()),
        }
        additions.extend([user, assistant])
        if parent_id:
            history[parent_id].setdefault("childrenIds", []).append(additions[0]["id"])
        for message in additions:
            history[message["id"]] = message
        chat["history"]["currentId"] = assistant_id
        chat["messages"] = list(history.values())
        chat["params"] = {**(chat.get("params") or {}), **params}
        chat["params"].pop("system", None)
        if "files" in body:
            chat["files"] = deepcopy(body["files"])
        endpoint = (
            f"/api/v1/chats/{state.chat_id}" if state.chat_id else "/api/v1/chats/new"
        )
        result = await self._async_webui_request(
            "POST", endpoint, state.headers, {"chat": chat}
        )
        state.chat_id = result["id"]
        state.assistant_id = assistant_id
        state.model_id = model["id"]
        state.text.clear()
        state.tool_calls.clear()
        state.tool_results.clear()
        body.update(
            chat_id=state.chat_id,
            id=assistant_id,
            user_message=user,
            session_id=self._webui_socket.get_sid(),
            stream=True,
        )
        return body

    @staticmethod
    def _webui_output_text(state: WebUIChat, event: dict) -> list[dict]:
        """Translate native text and external tool events into HA chat deltas."""
        data = event.get("data") or {}
        chunks = []
        if not isinstance(data, dict):
            return chunks
        if event.get("type") == "response:completion":
            kind = data.get("type", "")
            field_name = "thinking_content" if "reasoning" in kind else "content"
            if kind.endswith("text.delta") and isinstance(data.get("delta"), str):
                key = (
                    data["item_id"],
                    data.get("content_index", data.get("summary_index", 0)),
                    field_name,
                )
                state.text[key] = state.text.get(key, "") + data["delta"]
                chunks.append({field_name: data["delta"]})
        items = data.get("output") or []
        part_index = 0
        if event.get("type") == "response:completion":
            kind = data.get("type", "")
            if kind in ("response.completed", "response.incomplete"):
                # Provider completion is a snapshot, not the end of WebUI's turn.
                items = (data.get("response") or {}).get("output") or items
            elif (
                kind in ("response.output_text.done", "response.content_part.done")
                and data.get("item_id")
                and (kind != "response.content_part.done" or "content_index" in data)
            ):
                part = (
                    data.get("part") if kind == "response.content_part.done" else data
                )
                if (
                    isinstance(part, dict)
                    and isinstance(part.get("text"), str)
                    and (
                        kind == "response.output_text.done"
                        or part.get("type") == "output_text"
                    )
                ):
                    part_index = data.get("content_index", 0)
                    items = [
                        {
                            "id": data["item_id"],
                            "type": "message",
                            "content": [{"text": part["text"]}],
                        }
                    ]
        if (
            event.get("type") == "response:completion"
            and data.get("type") == "response.output_item.done"
            and (item := data.get("item", {})).get("type")
            in ("function_call", "function_call_output", "message")
        ):
            items = [item]
        for item in items:
            if item.get("type") == "function_call":
                call_id = item.get("call_id") or item["id"]
                if call_id in state.tool_calls or item.get("status") == "in_progress":
                    continue
                arguments = item.get("arguments") or "{}"
                try:
                    args = (
                        json.loads(arguments)
                        if isinstance(arguments, str)
                        else arguments
                    )
                except json.JSONDecodeError:
                    # Report the provider's call; WebUI handles invalid arguments.
                    args = {"json_decode_failure": arguments}
                state.tool_calls[call_id] = item["name"]
                chunks.append(
                    {
                        "tool_calls": [
                            llm.ToolInput(
                                id=call_id,
                                tool_name=item["name"],
                                tool_args=args,
                                external=True,
                            )
                        ]
                    }
                )
                continue
            if item.get("type") == "function_call_output":
                call_id = item["call_id"]
                if call_id in state.tool_results or call_id not in state.tool_calls:
                    continue
                state.tool_results.add(call_id)
                chunks.append(
                    {
                        "role": "tool_result",
                        "tool_call_id": call_id,
                        "tool_name": state.tool_calls[call_id],
                        "result": llm.ToolResult(data={"output": item.get("output")}),
                    }
                )
                continue
            if item.get("type") not in ("message", "reasoning"):
                continue
            field_name = (
                "thinking_content" if item["type"] == "reasoning" else "content"
            )
            parts = item.get("content") or item.get("summary") or []
            for index, part in enumerate(parts, start=part_index):
                text = part.get("text", "")
                key = (item["id"], index, field_name)
                previous = state.text.get(key, "")
                if (
                    field_name == "thinking_content"
                    and previous.strip() == text.strip()
                ):
                    # Native final snapshots trim reasoning whitespace.
                    # This content is never speech; do not repeat it or abort TTS.
                    state.text[key] = text
                elif text.startswith(previous):
                    if suffix := text[len(previous) :]:
                        chunks.append({field_name: suffix})
                    state.text[key] = text
                elif not previous.startswith(text):
                    msg = "OpenWebUI revised text that has already been spoken."
                    raise HomeAssistantError(msg)
        return chunks

    async def _async_webui_receive(
        self, state: WebUIChat, meta: GenerationMetadata
    ) -> AsyncGenerator[
        conversation.AssistantContentDeltaDict
        | conversation.ToolResultContentDeltaDict,
        None,
    ]:
        yield {"role": "assistant"}
        role = "assistant"
        while True:
            event = await state.events.get()
            if isinstance(event, BaseException):
                raise event
            if event.get("type") == "chat:tasks:cancel":
                msg = "OpenWebUI cancelled the response."
                raise HomeAssistantError(msg)
            data = event.get("data") or {}
            if not isinstance(data, dict):
                continue
            if event.get("type") == "chat:message:error" or data.get("error"):
                msg = f"OpenWebUI generation failed: {data.get('error', data)}"
                raise HomeAssistantError(msg)
            for chunk in self._webui_output_text(state, event):
                if not meta.time_to_first_token:
                    meta.time_to_first_token = round(
                        (time.time() - meta.request_start) * 1000, 2
                    )
                if chunk:
                    if chunk.get("role") == "tool_result":
                        role = "tool_result"
                    elif role != "assistant":
                        yield {"role": "assistant"}
                        role = "assistant"
                    yield chunk
            if (
                event.get("type") == "response:completion"
                and data.get("type") == "response.output_item.done"
                and (data.get("item") or {}).get("type") == "message"
                and (data.get("item") or {}).get("status") == "completed"
                and role == "assistant"
            ):
                # The native message is complete even when its text was already
                # emitted as deltas. Preserve that boundary for HA and reminders.
                yield {"role": "assistant"}
            if event.get("type") != "chat:completion":
                continue
            output = data.get("output") or []
            result_ids = {
                item.get("call_id")
                for item in output
                if item.get("type") == "function_call_output"
            }
            pending = next(
                (
                    item
                    for item in output
                    if item.get("type") == "function_call"
                    and item.get("status") in ("pending", "requires_approval")
                    and item.get("call_id", item.get("id")) not in result_ids
                ),
                None,
            )
            if pending is not None:
                state.pending = deepcopy(pending)
                if "call_id" not in state.pending:
                    state.pending["call_id"] = pending["id"]
                state.answers.clear()
                yield {"role": "assistant"}
                yield {"content": self._webui_question(state)}
                return
            if data.get("done") is True:
                if role == "tool_result":
                    # HA requires a final assistant message, including silent replies.
                    yield {"role": "assistant", "content": ""}
                meta.usage = data.get("usage")
                meta.tool_names = list(
                    dict.fromkeys(
                        item["name"]
                        for item in output
                        if item.get("type") == "function_call"
                    )
                )
                meta.finish_reason = "stop"
                return

    async def _async_webui_turn(  # noqa: C901 - Preserve the existing native turn lifecycle.
        self,
        chat_log: conversation.ChatLog,
        user_input: conversation.ConversationInput | None = None,
        *,
        structure_name: str | None = None,
        structure: vol.Schema | None = None,
        parallel_tool_calls: bool = False,
    ) -> bool:
        state = self._webui_chats.setdefault(chat_log.conversation_id, WebUIChat())
        meta = GenerationMetadata()
        previous_answers = dict(state.answers)
        # Reminders must never enter the request that they accompany.
        contents = list(chat_log.content)
        loop = asyncio.get_running_loop()
        # Leave room inside the client's 30 s HTTP deadline for STT and TTS.
        # Subsequent reminders remain less frequent while native work continues.
        deadline = loop.time() + 10.0

        async def native_stream() -> AsyncGenerator[dict]:
            if state.pending:
                resolution = self._webui_answer(
                    state, user_input.text if user_input else ""
                )
                if resolution is None:
                    yield {"role": "assistant", "content": self._webui_question(state)}
                    return
                context = AgentContext(conversation_id=chat_log.conversation_id)
                state.headers = await self._async_get_custom_headers(context)
            else:
                resolution = None
                model_args, context, injected = await self._async_prepare_chat_log(
                    chat_log,
                    contents,
                    structure_name,
                    structure,
                    user_input,
                    parallel_tool_calls=parallel_tool_calls,
                )
                self._get_extra_body_args(context)
                state.headers = model_args["extra_headers"]
            # Keep generation timing separate from the earlier audio deadline.
            meta.request_start = time.time()
            await self._async_webui_connect(state.headers)
            state.events = asyncio.Queue()
            if resolution is None:
                body = await self._async_webui_prepare_turn(
                    state, model_args, context.extra_body, injected
                )
                path = "/api/chat/completions"
            else:
                body = resolution
                path = f"/api/v1/chats/{state.chat_id}/messages/{state.assistant_id}/resolve"
            state.active = True
            accepted = await self._async_webui_request(
                "POST", path, state.headers, body
            )
            if not accepted.get("status"):
                msg = "OpenWebUI did not accept the request."
                raise HomeAssistantError(msg)
            if resolution is not None:
                state.pending = None
                state.answers.clear()
            async for chunk in self._transform_stream(
                self._async_webui_receive(state, meta),
                strip_emojis=self.options.get(CONF_STRIP_EMOJIS),
                conversation_id=chat_log.conversation_id,
                meta=meta,
                native_deltas=True,
            ):
                yield chunk

        async def progress_stream() -> AsyncGenerator[dict]:
            nonlocal deadline
            native = native_stream()
            pending = asyncio.create_task(anext(native))
            can_announce = True
            progress_open = False
            try:
                while True:
                    # Do not cancel native work when an announcement becomes due.
                    timeout = max(0.0, deadline - loop.time()) if can_announce else None
                    ready, _ = await asyncio.wait({pending}, timeout=timeout)
                    if ready:
                        try:
                            chunk = pending.result()
                        except StopAsyncIteration:
                            if progress_open:
                                yield {"role": "assistant", "content": ""}
                            return
                        if chunk.get("role") == "tool_result" or chunk.get(
                            "tool_calls"
                        ):
                            can_announce = True
                        elif chunk.get("content"):
                            # Content is not proof of audio. Keep the deadline;
                            # only prohibit interruption of an open model message.
                            can_announce = False
                        elif chunk.get("role") == "assistant":
                            can_announce = True
                        if progress_open and "role" not in chunk:
                            yield {"role": "assistant"}
                        progress_open = False
                        yield chunk
                        pending = asyncio.create_task(anext(native))
                        # Give an immediately available terminal/error event priority.
                        await asyncio.sleep(0)
                        if pending.done():
                            try:
                                next_chunk = pending.result()
                            except StopAsyncIteration:
                                if progress_open:
                                    yield {"role": "assistant", "content": ""}
                                return
                            if (
                                next_chunk.get("content")
                                or "role" in next_chunk
                                or next_chunk.get("tool_calls")
                            ):
                                continue
                        if not can_announce or loop.time() < deadline:
                            continue
                    if not progress_open:
                        yield {"role": "assistant"}
                        progress_open = True
                    yield {
                        "content": (
                            "Ich arbeite noch an deiner Anfrage. "
                            "Ich melde mich mit dem Ergebnis. "
                        )
                    }
                    deadline = loop.time() + 15.0
            finally:
                if not pending.done():
                    pending.cancel()
                await asyncio.gather(pending, return_exceptions=True)
                await native.aclose()

        try:
            progress = progress_stream()
            try:
                async for _ in chat_log.async_add_delta_content_stream(
                    self.entity_id, progress
                ):
                    pass
            finally:
                await progress.aclose()
            if state.pending is None:
                usage = meta.usage or {}
                self.hass.bus.async_fire(
                    "local_openai_llm_generation_complete",
                    {
                        "entity_id": self.entity_id,
                        "type": "generation_complete",
                        "model": state.model_id,
                        # One integration request; WebUI's internal loop stays upstream.
                        "iteration": 1,
                        "conversation_id": chat_log.conversation_id,
                        "content": next(
                            (
                                c.content
                                for c in reversed(chat_log.content)
                                if isinstance(c, conversation.AssistantContent)
                            ),
                            None,
                        ),
                        "tools_called": meta.tool_names or [],
                        "token_metadata": {
                            **usage,
                            "time_to_first_token_ms": meta.time_to_first_token,
                        },
                    },
                )
        except asyncio.CancelledError:
            if state.active and state.chat_id:
                try:
                    await asyncio.shield(self._async_webui_stop(state))
                except Exception:
                    _LOGGER.exception("OpenWebUI did not confirm cancellation")
            if state.pending:
                state.answers = previous_answers
            raise
        except (aiohttp.ClientError, socketio.exceptions.SocketIOError) as err:
            if state.pending:
                state.answers = previous_answers
            msg = f"OpenWebUI transport error: {err}"
            raise HomeAssistantError(msg) from err
        except Exception:
            if state.pending:
                state.answers = previous_answers
            raise
        finally:
            state.active = False
        return False

    def _get_extra_body_args(self, context: AgentContext) -> None:
        """
        Build extra_body args for the completion request.

        Chat Template Arguments are sent as a top-level `chat_template_kwargs`
        field, which most inference servers (llama.cpp, vLLM) read. Server-type
        subclasses may override this to deliver them elsewhere.
        """
        options = self.options
        extra_body_args = context.extra_body
        extra_body_args["stream_options"] = {
            "include_usage": True,
        }

        chat_template_opts = options.get(CONF_CHAT_TEMPLATE_OPTS, {})
        chat_template_args = chat_template_opts.get(CONF_CHAT_TEMPLATE_KWARGS, [])
        chat_template_args = [
            keypair for keypair in chat_template_args if keypair["Key"].strip()
        ]

        if chat_template_args:
            kwargs = {}
            for keypair in chat_template_args:
                if keypair["Key"]:
                    # Value is a template, so non-string types and structures can be provided
                    kwargs[keypair["Key"]] = template.Template(
                        keypair["Value"],
                        self.hass,
                    ).async_render()
            extra_body_args["chat_template_kwargs"] = kwargs

        request_body_opts = options.get(CONF_REQUEST_BODY_OPTS, {})
        request_body_parameters = request_body_opts.get(
            CONF_REQUEST_BODY_PARAMETERS, []
        )
        request_body_parameters = [
            keypair
            for keypair in request_body_parameters
            if keypair.get("Key", "").strip()
        ]

        for keypair in request_body_parameters:
            key = keypair.get("Key", "").strip()
            if key:
                extra_body_args[key] = template.Template(
                    keypair.get("Value", ""),
                    self.hass,
                ).async_render()

        if (
            self.server_options.get(CONF_PASS_SESSION_ID, False)
            and context.conversation_id
        ):
            extra_body_args.setdefault("metadata", {})["session_id"] = (
                context.conversation_id
            )

    async def _async_get_custom_headers(self, context: AgentContext) -> dict[str, str]:
        """Build custom headers dict with per-request Jinja2 template rendering."""
        custom_headers_section = self.entry.data.get(CONF_CUSTOM_HEADERS, {}) or {}
        custom_headers_list = custom_headers_section.get(CONF_SERVER_HEADERS, []) or []
        custom_headers = {item["Key"]: item["Value"] for item in custom_headers_list}
        result: dict[str, str] = {}
        for header_key, header_val in custom_headers.items():
            if isinstance(header_val, str) and "{" in header_val:
                try:
                    rendered_val = template.Template(
                        header_val, self.hass
                    ).async_render(
                        {"conversation_id": context.conversation_id},
                        parse_result=False,
                    )
                    result[header_key] = str(rendered_val)
                except Exception as err:
                    _LOGGER.warning(
                        "Error rendering custom header '%s': %s", header_key, err
                    )
                    result[header_key] = ""
            else:
                result[header_key] = str(header_val)
        return result

    async def _convert_content_to_chat_message(
        self,
        content: conversation.Content,
        conversation_id: str | None = None,
    ) -> ChatCompletionMessageParam | None:
        if isinstance(content, conversation.ToolResultContent):

            def log_and_str(value: Any) -> str:
                _LOGGER.warning(
                    "Attempting string convertion of non-JSON-serialisable response content from LLM tool '%s': %s",
                    content.tool_name,
                    value,
                )
                return str(value)

            return ChatCompletionToolMessageParam(
                role="tool",
                tool_call_id=content.tool_call_id,
                content=json.dumps(content.result.data, default=log_and_str),
            )

        role: Literal["user", "assistant", "system"] = content.role
        if role == "system" and content.content:
            return ChatCompletionSystemMessageParam(
                role="system",
                content=content.content,
            )

        if role == "user" and content.content:
            messages = []

            if content.attachments:
                loop = asyncio.get_running_loop()
                for attachment in content.attachments or ():
                    if not attachment.mime_type.startswith("image/"):
                        raise HomeAssistantError(
                            translation_domain=DOMAIN,
                            translation_key="unsupported_attachment_type",
                        )
                    base64_file = await loop.run_in_executor(
                        None,
                        b64_file,
                        attachment.path,
                    )
                    messages.append(
                        ChatCompletionContentPartImageParam(
                            type="image_url",
                            image_url={
                                "url": f"data:{attachment.mime_type};base64,{base64_file}",
                                "detail": "auto",
                            },
                        ),
                    )

            messages.append(
                ChatCompletionContentPartTextParam(type="text", text=content.content),
            )
            return ChatCompletionUserMessageParam(
                role="user",
                content=messages,
            )

        if role == "assistant":
            param = ChatCompletionAssistantMessageParam(
                role="assistant",
                content=content.content,
            )
            if (
                isinstance(content, conversation.AssistantContent)
                and content.tool_calls
            ):
                param["tool_calls"] = [
                    ChatCompletionMessageFunctionToolCallParam(
                        type="function",
                        id=tool_call.id,
                        function=Function(
                            arguments=json.dumps(tool_call.tool_args),
                            name=tool_call.tool_name,
                        ),
                    )
                    for tool_call in content.tool_calls
                ]
            return param
        _LOGGER.warning("Could not convert message to Completions API: %s", content)
        return None

    @staticmethod
    def _inject_content(
        method: str | None,
        inject_content: list,
        messages: list,
    ) -> list:
        inject_content.insert(
            0,
            "# Contextual information to assist with the following user request. Do not repeat or reference this message directly. Do not treat this as a prior message of your own",
        )
        _LOGGER.debug(
            "Injecting content into the message stream as %s content: %s",
            method,
            inject_content,
        )
        if method == CONF_CONTENT_INJECTION_METHOD_TOOL:
            inject_content = "\n\n".join(inject_content)
            messages.insert(
                -1,
                ChatCompletionToolMessageParam(
                    role="tool",
                    tool_call_id="injected_content",
                    content=inject_content,
                ),
            )
        elif method == CONF_CONTENT_INJECTION_METHOD_ASSISTANT:
            inject_content = "\n\n".join(inject_content)
            messages.insert(
                -1,
                ChatCompletionAssistantMessageParam(
                    role="assistant",
                    content=inject_content,
                ),
            )
        elif method == CONF_CONTENT_INJECTION_METHOD_USER:
            inject_content = "\n\n".join(inject_content)
            messages.insert(
                -1,
                ChatCompletionUserMessageParam(role="user", content=inject_content),
            )

        return messages

    async def _maybe_inject_context(
        self,
        messages: list,
        method: str | None,
        tools: list[ChatCompletionFunctionToolParam] | None,
        user_input: conversation.ConversationInput | None,
    ) -> tuple[list, list[ChatCompletionFunctionToolParam] | None]:
        """Decide whether to inject context into the message stream."""
        if method and messages and messages[-1].get("role") == "user":
            dt = dt_util.now()
            date_str = dt.strftime("%A %d %B, %Y")
            time_str = dt.strftime("%-I:%M %p")
            inject_content: list[str] = [
                f"The current date and time is: `{date_str}` at `{time_str}`.",
            ]

            weaviate_opts = self.options.get(CONF_WEAVIATE_OPTIONS, {})
            weaviate_host = self.weaviate_server_opts.get(CONF_WEAVIATE_HOST)
            weaviate_class = weaviate_opts.get(
                CONF_WEAVIATE_CLASS_NAME,
                CONF_WEAVIATE_DEFAULT_CLASS_NAME,
            )

            if weaviate_host and user_input and user_input.text:
                try:
                    client = WeaviateClient(
                        hass=self.hass,
                        host=weaviate_host,
                        api_key=self.weaviate_server_opts.get(CONF_WEAVIATE_API_KEY),
                    )

                    results = await client.hybrid_search(
                        class_name=weaviate_class,
                        query=user_input.text,
                        alpha=weaviate_opts.get(
                            CONF_WEAVIATE_HYBRID_SEARCH_ALPHA,
                            CONF_WEAVIATE_DEFAULT_HYBRID_SEARCH_ALPHA,
                        ),
                        threshold=weaviate_opts.get(
                            CONF_WEAVIATE_THRESHOLD,
                            CONF_WEAVIATE_DEFAULT_THRESHOLD,
                        ),
                        limit=int(
                            weaviate_opts.get(
                                CONF_WEAVIATE_MAX_RESULTS,
                                CONF_WEAVIATE_DEFAULT_MAX_RESULTS,
                            ),
                        ),
                    )

                    _LOGGER.debug("Weaviate results: %s", results)

                    result_content = [
                        f"Query: {result.get('query').strip()}\nContent: {result.get('content').strip()}"
                        for result in results
                    ]
                    if result_content:
                        inject_content += result_content
                except Exception:
                    _LOGGER.exception(
                        "An unexpected exception occurred while processing RAG",
                    )

            if inject_content:
                messages = self._inject_content(method, inject_content, messages)
                if tools:
                    tools = [
                        tool
                        for tool in tools
                        if not tool["function"]["name"].endswith("GetDateTime")
                    ]
        return messages, tools

    def _accumulate_tool_call(
        self,
        pending_tool_calls: dict,
        tool_call_id: str | None,
        tool_call_name: str | None,
        tool_call: ChatCompletionMessageToolCall,
        conversation_id: str | None = None,
    ) -> tuple[str | None, str | None]:
        """
        Accumulate a tool call delta into pending_tool_calls.

        Returns the (possibly updated) tool_call_id and tool_call_name.
        """
        # llama.cpp - only the initial tool call chunk has an ID, subsequent argument chunks do not
        # Ollama - parallel tool calls all share the same .index value (0)
        tool_call_id = tool_call.id or tool_call_id

        # And some mystery engine from OpenRouter uses the same index and ID across parallel tool requests within so lets track the tool name itself for changes as well
        tool_call_name = (
            tool_call.function.name
            if tool_call.function.name and tool_call.function.name != tool_call_name
            else tool_call_name
        )
        tool_key = tool_call_id + tool_call_name

        if tool_key not in pending_tool_calls:
            pending_tool_calls[tool_key] = {
                "id": tool_call_id,
                "name": tool_call.function.name,
                "args": tool_call.function.arguments or "",
            }
        else:
            pending_tool_calls[tool_key]["args"] += tool_call.function.arguments or ""

        self._on_tool_call_delta(
            tool_call_id, tool_call_name, tool_call, conversation_id
        )
        return tool_call_id, tool_call_name

    def _on_tool_call_delta(
        self,
        tool_call_id: str | None,
        tool_call_name: str | None,
        tool_call: ChatCompletionMessageToolCall,
        conversation_id: str | None = None,
    ) -> None:
        """
        Process tool call delta during streaming.

        Subclasses can override to capture additional data from the raw chunk.
        """

    async def _transform_stream(  # noqa: C901 todo: will break this up
        self,
        stream: AsyncStream[ChatCompletionChunk] | AsyncGenerator[dict, None],
        strip_emojis: bool,
        conversation_id: str | None = None,
        meta: GenerationMetadata | None = None,
        *,
        native_deltas: bool = False,
    ) -> AsyncGenerator[
        conversation.AssistantContentDeltaDict
        | conversation.ToolResultContentDeltaDict,
        None,
    ]:
        """Transform a streaming OpenAI response to ChatLog format."""
        new_msg = True
        pending_think = ""
        in_think = False
        seen_visible = False
        loop = asyncio.get_running_loop()
        pending_tool_calls = {}
        tool_call_id = None
        tool_call_name = None
        first_token_recorded = False

        async for event in stream:
            chunk: conversation.AssistantContentDeltaDict = {}

            if native_deltas:
                # WebUI already owns tool execution. Preserve HA's native deltas;
                # only assistant text goes through the original filter below.
                chunk = dict(event)
                choice = None
                if chunk.get("role") in ("assistant", "tool_result"):
                    pending_think = ""
                    in_think = False
                    seen_visible = False
                content = chunk.pop("content", None)
                if chunk.get("role") == "assistant" and content == "":
                    # HA distinguishes an explicit silent answer from role-only.
                    chunk["content"] = ""
                reasoning_content = chunk.pop("thinking_content", None)
            else:
                if not event.choices:
                    continue

                _LOGGER.debug(event)
                choice = event.choices[0]
                delta = choice.delta

                if not first_token_recorded and meta is not None and meta.request_start:
                    has_token = (
                        (delta.content is not None)
                        or (
                            getattr(delta, "reasoning_content", None)
                            or getattr(delta, "reasoning", None)
                        )
                        or (delta.tool_calls is not None and delta.tool_calls)
                    )
                    if has_token:
                        meta.time_to_first_token = round(
                            (time.time() - meta.request_start) * 1000, 2
                        )
                        first_token_recorded = True

                if new_msg:
                    # openvinotoolkit/model_server fails to provide a message role in its responses, so lets default to assistant if none is received
                    chunk["role"] = delta.role or "assistant"
                    new_msg = False

                if (tool_calls := delta.tool_calls) is not None and tool_calls:
                    # I've never seen this contain more than a single tool call, but let's iterate over it just in case
                    for tool_call in tool_calls:
                        tool_call_id, tool_call_name = self._accumulate_tool_call(
                            pending_tool_calls,
                            tool_call_id,
                            tool_call_name,
                            tool_call,
                            conversation_id,
                        )

                # Handle reasoning_content field (used by reasoning models via OpenAI-compatible APIs)
                # Naming for this field varies. See https://github.com/vllm-project/vllm/issues/27755
                reasoning_content = getattr(
                    delta, "reasoning_content", None
                ) or getattr(
                    delta,
                    "reasoning",
                    None,
                )
                content = delta.content
            if reasoning_content:
                if _SUPPORTS_THINKING:
                    chunk["thinking_content"] = reasoning_content
                else:
                    _LOGGER.debug("LLM Thought: %s", reasoning_content)

            if content is not None:
                if strip_emojis:
                    content = await loop.run_in_executor(
                        None,
                        demoji.replace,
                        content,
                        "",
                    )

                # Handle <think> tags that may appear within larger chunks
                # (not just as exact token matches)
                if "<think>" in content:
                    in_think = True
                    content = content.replace("<think>", "")
                    pending_think = ""

                if in_think:
                    if "</think>" in content:
                        in_think = False
                        before_close, remaining = content.split("</think>", 1)
                        pending_think += before_close
                        content = remaining

                        if _SUPPORTS_THINKING:
                            # HA 2026.4+: stream the final thinking fragment
                            if before_close:
                                chunk["thinking_content"] = before_close
                        elif pending_think.strip():
                            _LOGGER.debug("LLM Thought: %s", pending_think)

                        pending_think = ""
                    else:
                        pending_think += content
                        if _SUPPORTS_THINKING:
                            # HA 2026.4+: stream thinking content as it arrives
                            chunk["thinking_content"] = content
                        content = ""

                if not in_think and content.strip():
                    seen_visible = True

                if seen_visible:
                    chunk["content"] = content

            tool_error_deltas: list[dict] = []
            if choice is not None and choice.finish_reason:
                # Capture usage stats for event emission
                meta.usage = getattr(event, "usage", None)
                meta.timings = getattr(event, "timings", None)
                meta.finish_reason = choice.finish_reason

                # Capture tool names before they're processed/cleared
                if meta is not None and pending_tool_calls:
                    meta.tool_names = [tc["name"] for tc in pending_tool_calls.values()]

                if pending_tool_calls:
                    parsed_tool_calls: list[llm.ToolInput] = []
                    for tool_call in pending_tool_calls.values():
                        try:
                            args = (
                                json.loads(tool_call["args"])
                                if tool_call["args"]
                                else {}
                            )
                            parsed_tool_calls.append(
                                llm.ToolInput(
                                    id=tool_call["id"],
                                    tool_name=tool_call["name"],
                                    tool_args=args,
                                    external=False,
                                )
                            )
                        except json.JSONDecodeError:  # noqa: PERF203
                            parsed_tool_calls.append(
                                llm.ToolInput(
                                    id=tool_call["id"],
                                    tool_name=tool_call["name"],
                                    tool_args={
                                        "json_decode_failure": tool_call["args"]
                                    },
                                    external=True,
                                )
                            )
                            tool_error_deltas.append(
                                {
                                    "role": "tool_result",
                                    "tool_call_id": tool_call["id"],
                                    "tool_name": tool_call["name"],
                                    "tool_result": {
                                        "error": f"Failed to parse tool arguments: {tool_call['args']}\n\nCheck your JSON and try again.",
                                    },
                                }
                            )

                    _LOGGER.debug("Calling tools: %s", parsed_tool_calls)
                    chunk["tool_calls"] = parsed_tool_calls
                    pending_tool_calls.clear()

            if (
                seen_visible
                or chunk.get("tool_calls")
                or chunk.get("role")
                or chunk.get("thinking_content")
            ):
                yield chunk
                for error_delta in tool_error_deltas:
                    yield error_delta

    async def _async_handle_chat_log(
        self,
        chat_log: conversation.ChatLog,
        structure_name: str | None = None,
        structure: vol.Schema | None = None,
        user_input: conversation.ConversationInput | None = None,
        *,
        parallel_tool_calls: bool = False,
    ) -> bool:
        if self._uses_webui:
            return await self._async_webui_turn(
                chat_log,
                user_input=user_input,
                structure_name=structure_name,
                structure=structure,
                parallel_tool_calls=parallel_tool_calls,
            )
        model_args, context, _ = await self._async_prepare_chat_log(
            chat_log,
            list(chat_log.content),
            structure_name,
            structure,
            user_input,
            parallel_tool_calls=parallel_tool_calls,
        )
        return await self._run_agent_loop(
            self.entry.runtime_data, model_args, chat_log, context
        )

    async def _async_prepare_chat_log(  # noqa: PLR0913 - Keep the existing request arguments.
        self,
        chat_log: conversation.ChatLog,
        contents: list,
        structure_name: str | None,
        structure: vol.Schema | None,
        user_input: conversation.ConversationInput | None,
        *,
        parallel_tool_calls: bool,
    ) -> tuple[dict, AgentContext, dict | None]:
        """Prepare existing request arguments from the pre-announcement input."""
        options = self.options
        max_message_history = options.get(CONF_MAX_MESSAGE_HISTORY)
        if max_message_history is not None:
            max_message_history = int(max_message_history)

        model_args = {
            "model": await self._async_get_model(chat_log),
            "parallel_tool_calls": parallel_tool_calls,
            "extra_headers": {
                "HTTP-Referer": "https://github.com/skye-harris/hass_local_openai_llm",
                "X-Title": "Home Assistant",
            },
        }

        context = AgentContext(conversation_id=chat_log.conversation_id)

        # Merge custom HTTP headers (with per-request Jinja2 template rendering)
        custom_headers = await self._async_get_custom_headers(context)
        for header_key, header_val in custom_headers.items():
            if header_key not in model_args["extra_headers"]:
                model_args["extra_headers"][header_key] = header_val

        # Omit when unset, some providers (e.g. Anthropic) reject a non-default temperature.
        temperature = options.get(CONF_TEMPERATURE)
        if temperature is not None:
            model_args["temperature"] = temperature

        tools: list[ChatCompletionFunctionToolParam] | None = None
        if chat_log.llm_api and not self._uses_webui:
            # Sorted by name because intent.async_get() yields tools in registration
            # order, which differs from one restart to the next. Tool schemas render
            # at the start of the prompt, so a reorder invalidates the whole prefix
            # of a server-side prompt cache.
            tools = [
                _format_tool(tool, chat_log.llm_api.custom_serializer)
                for tool in sorted(chat_log.llm_api.tools, key=lambda tool: tool.name)
            ]

        messages = self._trim_history(
            [
                m
                for content in contents
                if not (
                    self._uses_webui
                    and not options.get(CONF_OPENWEBUI_ATTACH_HA_PROMPT, False)
                    and content.role == "system"
                )
                if (
                    m := await self._convert_content_to_chat_message(
                        content, chat_log.conversation_id
                    )
                )
            ],
            max_message_history,
        )

        method = options.get(CONF_CONTENT_INJECTION_METHOD)
        if self._uses_webui and method == CONF_CONTENT_INJECTION_METHOD_TOOL:
            method = None
        original_message_count = len(messages)
        messages, tools = await self._maybe_inject_context(
            messages,
            method,
            tools,
            user_input,
        )
        model_args["messages"] = messages

        if tools:
            model_args["tools"] = tools

        if structure:
            if TYPE_CHECKING:
                assert structure_name is not None
            model_args["response_format"] = ResponseFormatJSONSchema(
                type="json_schema",
                json_schema=_format_structured_output(
                    structure_name,
                    structure,
                    chat_log.llm_api,
                ),
            )

        injected = messages[-2] if len(messages) > original_message_count else None
        return model_args, context, injected

    async def _run_agent_loop(
        self,
        client: openai.AsyncOpenAI,
        model_args: dict[str, Any],
        chat_log: conversation.ChatLog,
        context: AgentContext,
    ) -> bool:
        """Run the LLM agent loop with tool call iteration."""
        halt_requested = False
        strip_emojis = self.options.get(CONF_STRIP_EMOJIS)

        for iteration in range(MAX_TOOL_ITERATIONS):
            context.iteration = iteration
            self._get_extra_body_args(context)
            if context.extra_body:
                model_args["extra_body"] = context.extra_body
            else:
                model_args.pop("extra_body", None)
            _LOGGER.debug("Extra-body args: %s", context.extra_body)

            if iteration == MAX_TOOL_ITERATIONS - 1:
                _LOGGER.debug("Max iterations reached, injecting stop message")
                model_args["messages"] = self._inject_stop_directive(
                    model_args["messages"]
                )
                model_args["tool_choice"] = "none"

            meta: GenerationMetadata = GenerationMetadata(request_start=time.time())

            try:
                result_stream = await client.chat.completions.create(
                    **model_args,
                    stream=True,
                )

                turn_messages = [
                    content
                    async for content in chat_log.async_add_delta_content_stream(
                        self.entity_id,
                        self._transform_stream(
                            stream=result_stream,
                            strip_emojis=strip_emojis,
                            conversation_id=context.conversation_id,
                            meta=meta,
                        ),
                    )
                ]

                # Check for stop_talking tool call before merging into message chain
                if not halt_requested:
                    for content in turn_messages:
                        if (
                            isinstance(content, conversation.AssistantContent)
                            and content.tool_calls
                        ):
                            halt_requested = any(
                                tool
                                for tool in content.tool_calls
                                if tool.tool_name == TOOL_NAME_STOP_TALKING
                            )
                            break

                model_args["messages"].extend(
                    [
                        msg
                        for content in turn_messages
                        if (
                            msg := await self._convert_content_to_chat_message(
                                content, context.conversation_id
                            )
                        )
                    ],
                )

                # Find the last assistant content added by this generation
                content_text: str | None = None
                for content in reversed(chat_log.content):
                    if isinstance(content, conversation.AssistantContent):
                        content_text = content.content or None
                        break

                # Fire event with all collected metadata
                usage_dict: dict[str, Any] | None = None
                if meta.usage is not None:
                    usage_dict = {
                        "prompt_tokens": meta.usage.prompt_tokens,
                        "completion_tokens": meta.usage.completion_tokens,
                        "total_tokens": meta.usage.total_tokens,
                        "completion_tokens_details": (
                            meta.usage.completion_tokens_details
                        ),
                        "prompt_tokens_details": meta.usage.prompt_tokens_details,
                    }

                timings_dict: dict[str, Any] = {}
                if meta.timings is not None:
                    if hasattr(meta.timings, "keys") and hasattr(meta.timings, "get"):
                        for key in meta.timings:
                            timings_dict[key] = meta.timings.get(key)
                    else:
                        timings_dict.update(
                            {
                                k: v
                                for k, v in meta.timings.__dict__.items()
                                if not k.startswith("_")
                            }
                        )

                self.hass.bus.async_fire(
                    "local_openai_llm_generation_complete",
                    {
                        "entity_id": self.entity_id,
                        "type": "generation_complete",
                        "model": model_args["model"],
                        "iteration": iteration + 1,
                        "conversation_id": context.conversation_id,
                        "content": content_text,
                        "tools_called": meta.tool_names or [],
                        "token_metadata": {
                            **(usage_dict or timings_dict),
                            "time_to_first_token_ms": meta.time_to_first_token,
                        },
                    },
                )

            except openai.OpenAIError as err:
                _LOGGER.exception("API server returned an error")
                msg = "API server returned an error. Check the system logs for further details."
                raise HomeAssistantError(msg) from err
            except Exception as err:
                _LOGGER.exception("Error handling API response")
                msg = "Error handling API response. Check the system logs for further details."
                raise HomeAssistantError(msg) from err

            if not chat_log.unresponded_tool_results:
                break

        return halt_requested

    @staticmethod
    def _inject_stop_directive(messages: list) -> list:
        """Append a stop-directive to the last tool result if present."""
        if not messages:
            return messages
        last_msg = messages[-1]
        if last_msg.get("role") != "tool":
            return messages
        directive = (
            "\n\nYou have reached the maximum number of tool call iterations. "
            "You must now provide your final response directly to the user "
            "without calling any more tools."
        )
        existing_content = last_msg.get("content", "")
        last_msg["content"] = existing_content + directive
        return messages

    @staticmethod
    def _trim_history(messages: list, max_messages: int | None) -> list:
        """
        Trim history to a configurable number of previous conversation rounds.

        Groups messages into rounds by user boundaries so tool-call rounds
        (assistant→tool→tool→user) stay intact.
        """
        if max_messages is None or max_messages < 0:
            return messages

        if len(messages) <= 1:
            return messages

        system = messages[0]
        rest = messages[1:]

        rounds: list[list] = []
        current_round: list = []
        for msg in rest:
            if msg["role"] == "user":
                if current_round:
                    rounds.append(current_round)
                current_round = [msg]
            else:
                current_round.append(msg)
        if current_round:
            rounds.append(current_round)

        if not rounds:
            return messages

        num_previous = len(rounds) - 1
        if num_previous >= max_messages:
            rounds = (
                [rounds[-1]] if max_messages == 0 else rounds[-(max_messages + 1) :]
            )

        return [system, *[msg for r in rounds for msg in r]]

    async def upsert_data_in_weaviate(
        self,
        query: str,
        content: str,
        identifier: str | None,
    ) -> None:
        """Add or update a record in Weaviate."""
        weaviate_opts = self.options.get(CONF_WEAVIATE_OPTIONS, {})
        weaviate_host = self.weaviate_server_opts.get(CONF_WEAVIATE_HOST)
        weaviate_class = weaviate_opts.get(
            CONF_WEAVIATE_CLASS_NAME,
            CONF_WEAVIATE_DEFAULT_CLASS_NAME,
        )

        if not weaviate_host:
            msg = "Weaviate is not configured for this Agent"
            raise RuntimeError(msg)

        client = WeaviateClient(
            hass=self.hass,
            host=weaviate_host,
            api_key=self.weaviate_server_opts.get(CONF_WEAVIATE_API_KEY),
        )

        # If we have been provided an identifier, generate a UUID and check if it exists
        object_uuid = _make_uuid(identifier) if identifier else None
        if object_uuid:
            object_exists = await client.does_object_exist(
                class_name=weaviate_class,
                object_uuid=object_uuid,
            )

            if object_exists:
                # Object exists, perform replace the existing object with the new one
                await client.replace_object(
                    class_name=weaviate_class,
                    query=query,
                    content=content,
                    object_uuid=object_uuid,
                )

                _LOGGER.info("Object updated in Weaviate: %s", object_uuid)
                return

        # Object does not exist, create new object
        await client.add_object(
            class_name=weaviate_class,
            query=query,
            content=content,
            object_uuid=object_uuid,
        )

        _LOGGER.info("Object added to Weaviate class: %s", weaviate_class)
