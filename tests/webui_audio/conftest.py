"""Isolated intent-to-TTS probe; no STT or native DSP is exercised."""

import sys
from types import ModuleType
from unittest.mock import MagicMock
import pytest

sys.modules["turbojpeg"] = MagicMock()


class UnusedDSP:
    def __init__(self, *args, **kwargs):
        pass

    def __getattr__(self, name):
        raise AssertionError("No native DSP in intent-to-TTS probe")


for module_name, class_name in [
    ("pymicro_vad", "MicroVad"),
    ("pyspeex_noise", "AudioProcessor"),
]:
    module = ModuleType(module_name)
    setattr(module, class_name, UnusedDSP)
    sys.modules[module_name] = module


@pytest.fixture(autouse=True)
def installed_dependencies(hass):
    hass.config.skip_pip = True
