"""Common variables used in the Ultimate RVC project."""

from __future__ import annotations

from typing import TYPE_CHECKING

import importlib.util
import os
import sys
from pathlib import Path

if TYPE_CHECKING:
    from types import ModuleType

BASE_DIR = Path.cwd()
MODELS_DIR = Path(os.getenv("URVC_MODELS_DIR") or BASE_DIR / "models")
RVC_MODELS_DIR = MODELS_DIR / "rvc"
VOICE_MODELS_DIR = Path(
    os.getenv("URVC_VOICE_MODELS_DIR") or RVC_MODELS_DIR / "voice_models",
)
CUSTOM_EMBEDDER_MODELS_DIR = RVC_MODELS_DIR / "embedders" / "custom"
CUSTOM_PRETRAINED_MODELS_DIR = RVC_MODELS_DIR / "pretraineds"
SEPARATOR_MODELS_DIR = MODELS_DIR / "audio_separator"
TRAINING_MODELS_DIR = RVC_MODELS_DIR / "training"
AUDIO_DIR = Path(os.getenv("URVC_AUDIO_DIR") or BASE_DIR / "audio")
TEMP_DIR = Path(os.getenv("URVC_TEMP_DIR") or BASE_DIR / "temp")


def lazy_import(name: str) -> ModuleType:
    """
    Lazy import a module.

    Parameters
    ----------
    name : str
        The name of the module to import.

    Returns
    -------
    ModuleType
        The imported module.

    Raises
    ------
    ModuleNotFoundError
        If the module is not found.
    ImportError
        If the loader is not found for the module.

    """
    spec = importlib.util.find_spec(name)
    if spec is None:
        msg = f"module {name!r} not found"
        raise ModuleNotFoundError(msg)
    if spec.loader is None:
        msg = f"loader not found for module {name!r}"
        raise ImportError(msg)
    loader = importlib.util.LazyLoader(spec.loader)
    spec.loader = loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    loader.exec_module(module)

    return module
