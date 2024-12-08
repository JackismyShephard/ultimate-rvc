"""
Extra type definitions for the audio generation commands in the
CLI of the Ultimate RVC project.
"""

from __future__ import annotations

from enum import StrEnum


class PanelName(StrEnum):
    """
    Valid panel names for audio generation commands in the CLI of
    the Ultimate RVC project.
    """

    MAIN_OPTIONS = "Main Options"
    RVC_MAIN_OPTIONS = "RVC Main Options"
    EDGE_TTS_OPTIONS = "Edge TTS Options"
    VOICE_SYNTHESIS_OPTIONS = "Voice Synthesis Options"
    RVC_SYNTHESIS_OPTIONS = "RVC Synthesis Options"
    RVC_VOICE_SYNTHESIS_OPTIONS = "RVC Voice Synthesis Options"
    VOICE_ENRICHMENT_OPTIONS = "Voice Enrichment Options"
    VOCAL_ENRICHMENT_OPTIONS = "Vocal Enrichment Options"
    RVC_ENRICHMENT_OPTIONS = "RVC Enrichment Options"
    SPEAKER_EMBEDDINGS_OPTIONS = "Speaker Embeddings Options"
    RVC_EMBEDDINGS_OPTIONS = "RVC Embeddings Options"
    VOCAL_POST_PROCESSING_OPTIONS = "Vocal Post-processing Options"
    AUDIO_MIXING_OPTIONS = "Audio Mixing Options"
