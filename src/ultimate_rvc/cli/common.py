"""Common utilities for the CLI."""

from ultimate_rvc.typing_extra import (
    AudioExt,
    AudioNormalizationMode,
    AudioSplitMethod,
    DeviceType,
    EmbedderModel,
    F0Method,
    IndexAlgorithm,
    PrecisionType,
    PretrainedType,
    SampleRate,
    TrainingSampleRate,
    Vocoder,
)


def format_duration(seconds: float) -> str:
    """
    Format a duration in seconds to a human-readable string.

    Parameters
    ----------
    seconds : float
        The duration in seconds.

    Returns
    -------
    str
        The formatted duration

    """
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours > 0:
        return f"{int(hours)} hours, {int(minutes)} minutes, and {seconds:.2f} seconds"
    if minutes > 0:
        return f"{int(minutes)} minutes and {seconds:.2f} seconds"
    return f"{seconds:.2f} seconds"


def complete_name(incomplete: str, enumeration: list[str]) -> list[str]:
    """
    Return a list of names that start with the incomplete string.

    Parameters
    ----------
    incomplete : str
        The incomplete string to complete.
    enumeration : list[str]
        The list of names to complete from.

    Returns
    -------
    list[str]
        The list of names that start with the incomplete string.

    """
    return [name for name in list(enumeration) if name.startswith(incomplete)]


def complete_audio_ext(incomplete: str) -> list[str]:
    """
    Return a list of audio extensions that start with the incomplete
    string.

    Parameters
    ----------
    incomplete : str
        The incomplete string to complete.

    Returns
    -------
    list[str]
        The list of audio extensions that start with the incomplete
        string.

    """
    return complete_name(incomplete, list(AudioExt))


def complete_f0_method(incomplete: str) -> list[str]:
    """
    Return a list of F0 methods that start with the incomplete string.

    Parameters
    ----------
    incomplete : str
        The incomplete string to complete.

    Returns
    -------
    list[str]
        The list of F0 methods that start with the incomplete string.

    """
    return complete_name(incomplete, list(F0Method))


def complete_embedder_model(incomplete: str) -> list[str]:
    """
    Return a list of embedder models that start with the incomplete
    string.

    Parameters
    ----------
    incomplete : str
        The incomplete string to complete.

    Returns
    -------
    list[str]
        The list of embedder models that start with the incomplete
        string.

    """
    return complete_name(incomplete, list(EmbedderModel))


def complete_audio_split_method(incomplete: str) -> list[str]:
    """
    Return a list of audio split methods that start with the incomplete
    string.

    Parameters
    ----------
    incomplete : str
        The incomplete string to complete.

    Returns
    -------
    list[str]
        The list of audio split methods that start with the incomplete
        string.

    """
    return complete_name(incomplete, list(AudioSplitMethod))


def complete_sample_rate(incomplete: str) -> list[str]:
    """
    Return a list of sample rates that start with the incomplete string.

    Parameters
    ----------
    incomplete : str
        The incomplete string to complete.

    Returns
    -------
    list[str]
        The list of sample rates that start with the incomplete string.

    """
    return complete_name(incomplete, [str(sr) for sr in SampleRate])


def complete_training_sample_rate(incomplete: str) -> list[str]:
    """
    Return a list of training sample rates that start with the
    incomplete string.

    Parameters
    ----------
    incomplete : str
        The incomplete string to complete.

    Returns
    -------
    list[str]
        The list of training sample rates that start with the incomplete
        string.

    """
    return complete_name(incomplete, [str(sr) for sr in TrainingSampleRate])


def complete_normalization_mode(incomplete: str) -> list[str]:
    """
    Return a list of audio normalization modes that start with the
    incomplete string.

    Parameters
    ----------
    incomplete : str
        The incomplete string to complete.

    Returns
    -------
    list[str]
        The list of audio normalization modes that start with the
        incomplete string.

    """
    return complete_name(incomplete, list(AudioNormalizationMode))


def complete_vocoder(incomplete: str) -> list[str]:
    """
    Return a list of vocoders that start with the incomplete string.

    Parameters
    ----------
    incomplete : str
        The incomplete string to complete.

    Returns
    -------
    list[str]
        The list of vocoders that start with the incomplete string.

    """
    return complete_name(incomplete, list(Vocoder))


def complete_index_algorithm(incomplete: str) -> list[str]:
    """
    Return a list of index algorithms that start with the incomplete
    string.

    Parameters
    ----------
    incomplete : str
        The incomplete string to complete.

    Returns
    -------
    list[str]
        The list of index algorithms that start with the incomplete
        string.

    """
    return complete_name(incomplete, list(IndexAlgorithm))


def complete_device_type(incomplete: str) -> list[str]:
    """
    Return a list of device types that start with the incomplete
    string.

    Parameters
    ----------
    incomplete : str
        The incomplete string to complete.

    Returns
    -------
    list[str]
        The list of device types that start with the incomplete
        string.

    """
    return complete_name(incomplete, list(DeviceType))


def complete_precision_type(incomplete: str) -> list[str]:
    """
    Return a list of precision types that start with the incomplete
    string.

    Parameters
    ----------
    incomplete : str
        The incomplete string to complete.

    Returns
    -------
    list[str]
        The list of precision types that start with the incomplete
        string.

    """
    return complete_name(incomplete, list(PrecisionType))


def complete_pretrained_type(incomplete: str) -> list[str]:
    """
    Return a list of pretrained model types that start with the
    incomplete string.

    Parameters
    ----------
    incomplete : str
        The incomplete string to complete.

    Returns
    -------
    list[str]
        The list of pretrained model types that start with the
        incomplete string.

    """
    return complete_name(incomplete, list(PretrainedType))
