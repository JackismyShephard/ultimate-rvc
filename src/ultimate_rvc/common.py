"""Common variables used in the Ultimate RVC project."""

from __future__ import annotations

import os
from pathlib import Path

BASE_DIR = Path.cwd()
MODELS_DIR = Path(os.getenv("URVC_MODELS_DIR") or BASE_DIR / "models")
RVC_MODELS_DIR = MODELS_DIR / "rvc"
VOICE_MODELS_DIR = Path(
    os.getenv("URVC_VOICE_MODELS_DIR") or RVC_MODELS_DIR / "voice_models",
)
EMBEDDER_MODELS_DIR = RVC_MODELS_DIR / "embedders"
CUSTOM_EMBEDDER_MODELS_DIR = EMBEDDER_MODELS_DIR / "custom"
PRETRAINED_MODELS_DIR = RVC_MODELS_DIR / "pretraineds"
CUSTOM_PRETRAINED_MODELS_DIR = PRETRAINED_MODELS_DIR / "custom"

SEPARATOR_MODELS_DIR = MODELS_DIR / "audio_separator"
TRAINING_MODELS_DIR = RVC_MODELS_DIR / "training"
AUDIO_DIR = Path(os.getenv("URVC_AUDIO_DIR") or BASE_DIR / "audio")
TEMP_DIR = Path(os.getenv("URVC_TEMP_DIR") or BASE_DIR / "temp")
CONFIG_DIR = Path(os.getenv("URVC_CONFIG_DIR") or BASE_DIR / "config")
