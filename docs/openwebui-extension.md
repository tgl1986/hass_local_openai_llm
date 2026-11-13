# OpenWebUI extension on upstream 1.15.1

This fork integrates the existing native OpenWebUI extension with upstream
`dab61de01e3a7c906c404a27338afda92b1ab5f3` (1.15.1).
The extension was developed against 1.15.0 (`de73dc3812ef269f02f315ada6dd689d536ff1b2`),
including the prompt/tools update, pipeline timeout patch, voice streaming fix,
and the subsequently approved 10-second standby adjustment. These changes are
already in the source tree; do not reapply the old 1.15.0 patches to this fork.

## Behaviour and boundaries

- Enable the native OpenWebUI option for the relevant agent. Configure its server
  and credentials through the existing integration configuration. WebUI retains
  ownership of Workspace prompts, permissions, tools and tool execution.
- HA prompt attachment is optional. Pending questions, resolution and cancellation
  use the existing WebUI conversation endpoints and saved chat state.
- Native events enter HA ChatLog through the integration's existing streaming
  transformer, including its think and optional emoji filtering. There is no
  additional Markdown parser, sentence buffer or replacement TTS implementation.
- Completed response items are forwarded promptly with existing deduplication;
  tool boundaries reset thinking state. Harmless reasoning whitespace changes
  do not abort a turn.
- A standby message becomes eligible after 10 seconds from native turn-handler
  entry, then every 15 seconds after a reminder while work continues. An open
  model message is not interrupted. The message is:
  `Ich arbeite noch an deiner Anfrage. Ich melde mich mit dem Ergebnis. `
  It uses ordinary HA text/TTS streaming. These intervals are not audio playback
  deadlines and do not change Companion's independent 30-second HTTP timeout.
- The existing configurable Assist pipeline timeout registration is preserved:
  the integration delegates to the original WebSocket command handler with the
  selected agent's timeout. It does not edit HA Core or alter already active runs.
- HA Core, Companion, WebUI and TTS are not modified. Non-native SDK agents retain
  their existing request and tool paths.

## Upstream merge decisions

The 1.15.1 vLLM migration default (zero thinking budget), llama.cpp FIRST_LOOP
budget behaviour and minimum HA version 2026.11.0 are retained. The upstream
StopTalkingTool duplicate guard is combined with the extension's native-WebUI
exclusion. Manifest version stays 1.15.1 and adds the existing Socket.IO dependency.
Upstream `entity.py` was unchanged between 1.15.0 and 1.15.1. The imported extension
file had SHA256 `799d8fd7cf27662c5a972e9c175edb8f9a62dd0f74ae10f4938667e47a9d68bb`
before repository formatting and lint-only adjustments.

## Validation

Run the existing project checks with Python 3.14 and `requirements_test.txt`:

```sh
python -m pytest -q --no-cov
python -m ruff check .
python -m ruff format . --check
```

Tests include upstream SDK behaviour, native events, pending/resolve/stop,
streaming, standby timing and pipeline timeout lifecycle. Audio replay uses
recorded fixture events and generated WAV data, not actual speech or live models.
The initial complete run found one test output path tied to the old sandbox;
it now uses pytest's temporary directory. No model run or deployment is part of
this repository integration. Owner runtime acceptance of the 10-second standby
adjustment remains separate from automated regression testing.

Final validation on Braincore: **454 passed**, 5 dependency deprecation warnings,
70.79 seconds. Ruff check and format check passed (66 Python files). An AST
comparison confirmed that the imported streaming implementation differs only by
formatting, explicit built-in import, type annotations and an equivalent exception
message assignment needed by the repository's existing lint rules.

## Native system-message handoff fix

The native WebUI path now omits HA system messages before Completions conversion
when HA prompt attachment is disabled. Previously it converted the empty initial
HA system message first, producing `Could not convert message to Completions API`,
and removed system messages later when building the WebUI request. Prompt attachment
and ordinary SDK agents retain their conversion paths; logging is unchanged.

Regression coverage uses the native `async_accept_pipeline_from_satellite` entry,
real HA STT/pipeline/ChatLog/TTS handling and the integration's complete native turn.
Only hardware, recognizer output and backend transport are fixtures. Both TTS modes
produce downloadable audio and finish the pipeline; reporting playback completion
returns the simulated satellite to idle. A backend exception also completes as a
spoken HA error response. Four handoff cases cover empty/nonempty excluded system
messages, enabled HA prompt attachment and ordinary SDK conversion.
This is software-path coverage, not validation on the owner's physical satellite.

Validation: 461 tests passed (5 dependency deprecation warnings), including seven
new handoff/satellite cases. Ruff check and format check passed. The two affected
native handoff cases fail against the previous code and pass with this fix.

## WebUI text-message storage fix

HA converts even text-only input to a Completions content-parts list. The native
WebUI path now joins text-only parts into a string for the saved user message and
the completion request's `user_message`. This also covers an attached HA prompt.
Both `history.messages` and the saved `messages` list receive that same string,
which the WebUI chat renderer expects. The Completions `messages` payload and
ordinary SDK conversion are unchanged. No image conversion is added.

This fixes newly written text messages for the reported OpenWebUI 0.13.3 issue;
it does not migrate previously saved chats. Existing chat IDs, message links,
Workspace options, tool execution, streaming and cancellation are preserved.
Regression tests use real HA text conversion with mocked WebUI transport, covering
creation, continuation and HA prompt attachment. Three assertions fail on the
previous source. Sandbox validation does not call a real model or production API.

Validation on 2026-11-13: 463 tests passed in 70.14 seconds (five dependency
deprecation warnings); Ruff lint and format checks passed for all 67 Python files.
