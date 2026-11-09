"""
Set agent-specific timeouts before Home Assistant starts Assist runs.

This replaces the registered command in memory and delegates to its original
handler. It does not edit Core files or change runs that are already underway.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from homeassistant.components import websocket_api
from homeassistant.core import callback
from homeassistant.exceptions import HomeAssistantError
from homeassistant.setup import async_setup_component

if TYPE_CHECKING:
    from collections.abc import Callable

    from homeassistant.core import HomeAssistant

COMMAND = "assist_pipeline/run"
DATA_KEY = "local_openai.pipeline_start_timeout"
type Settings = Callable[[], tuple[str, float]]


class _StartTimeout:
    """Share one command wrapper between the loaded conversation agents."""

    def __init__(self, hass: HomeAssistant) -> None:
        """Keep the original handler and its validation schema."""
        # Import after async_setup_component has loaded its dependencies.
        from homeassistant.components.assist_pipeline.const import (  # noqa: PLC0415
            DEFAULT_PIPELINE_TIMEOUT,
        )
        from homeassistant.components.assist_pipeline.error import (  # noqa: PLC0415
            PipelineNotFound,
        )
        from homeassistant.components.assist_pipeline.pipeline import (  # noqa: PLC0415
            async_get_pipeline,
        )

        self.hass = hass
        self.original, self.schema = hass.data[websocket_api.DOMAIN][COMMAND]
        self.providers: dict[object, Settings] = {}

        @callback
        def handle(
            hass: HomeAssistant,
            connection: websocket_api.ActiveConnection,
            msg: dict[str, Any],
        ) -> None:
            try:
                pipeline = async_get_pipeline(hass, msg.get("pipeline"))
            except PipelineNotFound:
                self.original(hass, connection, msg)
                return

            for settings in self.providers.values():
                entity_id, minimum = settings()
                if pipeline.conversation_engine == entity_id:
                    effective = max(
                        msg.get("timeout", DEFAULT_PIPELINE_TIMEOUT), minimum
                    )
                    msg = {**msg, "timeout": effective}
                    break

            # The registered original already schedules its own async work.
            self.original(hass, connection, msg)

        self.handler = handle
        websocket_api.async_register_command(hass, COMMAND, handle, self.schema)

    @callback
    def register(self, settings: Settings) -> Callable[[], None]:
        """Register live entity settings and return an unload callback."""
        token = object()
        self.providers[token] = settings

        @callback
        def remove() -> None:
            self.providers.pop(token, None)
            if self.providers:
                return
            current = self.hass.data[websocket_api.DOMAIN].get(COMMAND)
            if current is not None and current[0] is self.handler:
                websocket_api.async_register_command(
                    self.hass, COMMAND, self.original, self.schema
                )
            if self.hass.data.get(DATA_KEY) is self:
                self.hass.data.pop(DATA_KEY)

        return remove


async def async_register_agent_timeout(
    hass: HomeAssistant, settings: Settings
) -> Callable[[], None]:
    """Apply a minimum timeout to the selected agent's new pipeline runs."""
    if not await async_setup_component(hass, "assist_pipeline", {}):
        msg = "Assist pipeline setup failed"
        raise HomeAssistantError(msg)
    if (state := hass.data.get(DATA_KEY)) is None:
        state = hass.data[DATA_KEY] = _StartTimeout(hass)
    return state.register(settings)
