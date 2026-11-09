"""Text-only Assist tests; fail if optional native audio codecs are used."""

import sys
from types import ModuleType
from unittest.mock import MagicMock

import pytest

sys.modules["turbojpeg"] = MagicMock()


class UnusedAudioCodec:
    def __init__(self, *args, **kwargs):
        pass

    def __getattr__(self, name):
        raise AssertionError("Audio codecs must not run in the text-pipeline probe")


for module_name, class_name in (
    ("pymicro_vad", "MicroVad"),
    ("pyspeex_noise", "AudioProcessor"),
):
    module = ModuleType(module_name)
    setattr(module, class_name, UnusedAudioCodec)
    sys.modules[module_name] = module


@pytest.fixture(autouse=True)
def no_dependency_install(hass):
    """Use installed sandbox dependencies."""
    hass.config.skip_pip = True
