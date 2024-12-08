from ultimate_rvc.cli.common import complete_name
from ultimate_rvc.typing_extra import AudioExt, EmbedderModel, F0Method


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
