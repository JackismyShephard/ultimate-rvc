"""
Module which defines functions that faciliatate song cover generation
using RVC.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated

import logging
import operator
import shutil
import time
from collections.abc import Sequence
from contextlib import suppress
from functools import cache
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from cyclopts import App, Group, Parameter, validators

from pydantic import ValidationError

import gradio as gr

from rich import print as rprint
from rich.panel import Panel
from rich.table import Table

from ultimate_rvc.common import SEPARATOR_MODELS_DIR, lazy_import
from ultimate_rvc.core.common import (
    INTERMEDIATE_AUDIO_BASE_DIR,
    OUTPUT_AUDIO_DIR,
    copy_file_safe,
    display_progress,
    format_duration,
    get_file_hash,
    json_dump,
    json_load,
    validate_url,
)
from ultimate_rvc.core.exceptions import (
    Entity,
    InvalidLocationError,
    Location,
    NotFoundError,
    NotProvidedError,
    UIMessage,
    YoutubeUrlError,
)
from ultimate_rvc.core.generate.common import (
    convert,
    get_unique_base_path,
    mix_audio,
    validate_all_exist,
    validate_exists,
    wavify,
)
from ultimate_rvc.core.generate.typing_extra import (
    EffectedVocalsMetaData,
    FileMetaData,
    MixedAudioType,
    PitchShiftMetaData,
    RVCAudioMetaData,
    SeparatedAudioMetaData,
    SongSourceType,
)
from ultimate_rvc.typing_extra import (
    AudioExt,
    EmbedderModel,
    F0Method,
    RVCContentType,
    SegmentSize,
    SeparationModel,
    StrPath,
)

if TYPE_CHECKING:

    import yt_dlp

    import pedalboard
    import pedalboard._pedalboard as pedalboard_
    import pedalboard.io as pedalboard_io
    import soundfile as sf
    import sox
    import static_ffmpeg
    import static_sox

    # NOTE the only reason this is imported here is so we can annotate
    # the return type of the _get_audio_separator function
    from audio_separator.separator import Separator

else:
    yt_dlp = lazy_import("yt_dlp")

    pedalboard = lazy_import("pedalboard")
    pedalboard_ = lazy_import("pedalboard._pedalboard")
    pedalboard_io = lazy_import("pedalboard.io")
    sf = lazy_import("soundfile")
    sox = lazy_import("sox")
    static_ffmpeg = lazy_import("static_ffmpeg")
    static_sox = lazy_import("static_sox")

logger = logging.getLogger(__name__)


arguments_group = Group.create_ordered("Arguments")
parameters_group = Group.create_ordered("Parameters")
main_options_group = Group.create_ordered("Main options")
voice_synthesis_options_group = Group.create_ordered("Voice synthesis options")
vocal_enrichment_options_group = Group.create_ordered("Vocal enrichment options")
speaker_embeddings_options_group = Group.create_ordered("Speaker embeddings options")
vocal_post_processing_options_group = Group.create_ordered(
    "Vocal post-processing options",
)
audio_mixing_options_group = Group.create_ordered("Audio mixing options")

app = App(
    name="song-cover",
    help="Generate song covers using RVC.",
    help_format="markdown",
    group_arguments=arguments_group,
    group_parameters=parameters_group,
)


@cache
def _get_audio_separator(
    output_dir: StrPath = INTERMEDIATE_AUDIO_BASE_DIR,
    output_format: str = AudioExt.WAV,
    segment_size: int = SegmentSize.SEG_256,
    sample_rate: int = 44100,
) -> Separator:
    static_ffmpeg.add_paths()
    static_sox.add_paths()
    from audio_separator.separator import Separator  # noqa: PLC0415

    """
    Get an audio separator.

    Parameters
    ----------
    output_dir : StrPath, default=INTERMEDIATE_AUDIO_BASE_DIR
        The directory to save the separated audio to.
    output_format : str, default=AudioExt.WAV
        The format to save the separated audio in.
    segment_size : int, default=SegmentSize.SEG_256
        The segment size to use for separation.
    sample_rate : int, default=44100
        The sample rate to use for separation.

    Returns
    -------
    Separator
        An audio separator.

    """
    return Separator(
        model_file_dir=SEPARATOR_MODELS_DIR,
        output_dir=output_dir,
        output_format=output_format,
        sample_rate=sample_rate,
        mdx_params={
            "hop_length": 1024,
            "segment_size": segment_size,
            "overlap": 0.001,
            "batch_size": 1,
            "enable_denoise": False,
        },
    )


def initialize_audio_separator(progress_bar: gr.Progress | None = None) -> None:
    """
    Initialize the audio separator by downloading the models it uses.

    Parameters
    ----------
    progress_bar : gr.Progress, optional
        Gradio progress bar to update.

    """
    audio_separator = _get_audio_separator()
    for i, separator_model in enumerate(SeparationModel):
        if not Path(SEPARATOR_MODELS_DIR / separator_model).is_file():
            display_progress(
                f"Downloading {separator_model}...",
                i / len(SeparationModel),
                progress_bar,
            )
            audio_separator.download_model_files(separator_model)


def _get_input_audio_path(directory: StrPath) -> Path | None:
    """
    Get the path to the input audio file in the provided directory, if
    it exists.

    The provided directory must be located in the root of the
    intermediate audio base directory.

    Parameters
    ----------
    directory : StrPath
        The path to a directory.

    Returns
    -------
    Path | None
        The path to the input audio file in the provided directory, if
        it exists.

    Raises
    ------
    NotFoundError
        If the provided path does not point to an existing directory.
    InvalidLocationError
        If the provided path is not located in the root of the
        intermediate audio base directory"

    """
    dir_path = Path(directory)

    if not dir_path.is_dir():
        raise NotFoundError(entity=Entity.DIRECTORY, location=dir_path)

    if dir_path.parent != INTERMEDIATE_AUDIO_BASE_DIR:
        raise InvalidLocationError(
            entity=Entity.DIRECTORY,
            location=Location.INTERMEDIATE_AUDIO_ROOT,
            path=dir_path,
        )
    # NOTE directory should never contain more than one element which
    # matches the pattern "00_*"
    return next(dir_path.glob("00_*"), None)


def _get_input_audio_paths() -> list[Path]:
    """
    Get the paths to all input audio files in the intermediate audio
    base directory.

    Returns
    -------
    list[Path]
        The paths to all input audio files in the intermediate audio
        base directory.

    """
    # NOTE if we later add .json file for input then
    # we need to exclude those here
    return list(INTERMEDIATE_AUDIO_BASE_DIR.glob("*/00_*"))


def get_named_song_dirs() -> list[tuple[str, str]]:
    """
    Get the names of all saved songs and the paths to the
    directories where they are stored.

    Returns
    -------
    list[tuple[str, Path]]
        A list of tuples containing the name of each saved song
        and the path to the directory where it is stored.

    """
    return sorted(
        [
            (
                path.stem.removeprefix("00_"),
                str(path.parent),
            )
            for path in _get_input_audio_paths()
        ],
        key=operator.itemgetter(0),
    )


def _get_model_name(
    effected_vocals_track: StrPath | None = None,
    song_dir: StrPath | None = None,
) -> str:
    """
    Infer the name of the voice model used for vocal conversion from a
    an effected vocals track in a given song directory.

    If a voice model name cannot be inferred, "Unknown" is returned.

    Parameters
    ----------
    effected_vocals_track : StrPath, optional
        The path to an effected vocals track.
    song_dir : StrPath, optional
        The path to a song directory.

    Returns
    -------
    str
        The name of the voice model used for vocal conversion.

    """
    model_name = "Unknown"
    if not (effected_vocals_track and song_dir):
        return model_name
    effected_vocals_path = Path(effected_vocals_track)
    song_dir_path = Path(song_dir)
    effected_vocals_json_path = song_dir_path / f"{effected_vocals_path.stem}.json"
    if not effected_vocals_json_path.is_file():
        return model_name
    effected_vocals_dict = json_load(effected_vocals_json_path)
    try:
        effected_vocals_metadata = EffectedVocalsMetaData.model_validate(
            effected_vocals_dict,
        )
    except ValidationError:
        return model_name
    converted_vocals_track_name = effected_vocals_metadata.vocals_track.name
    converted_vocals_json_path = song_dir_path / Path(
        converted_vocals_track_name,
    ).with_suffix(
        ".json",
    )
    if not converted_vocals_json_path.is_file():
        return model_name
    converted_vocals_dict = json_load(converted_vocals_json_path)
    try:
        converted_vocals_metadata = RVCAudioMetaData.model_validate(
            converted_vocals_dict,
        )
    except ValidationError:
        return model_name
    return converted_vocals_metadata.model_name


def get_song_cover_name(
    effected_vocals_track: StrPath | None = None,
    song_dir: StrPath | None = None,
    model_name: str | None = None,
) -> str:
    """
    Generate a suitable name for a cover of a song based on the name
    of that song and the voice model used for vocal conversion.

    If the path of an existing song directory is provided, the name
    of the song is inferred from that directory. If a voice model is not
    provided but the path of an existing song directory and the path of
    an effected vocals track in that directory are provided, then the
    voice model is inferred from the effected vocals track.

    Parameters
    ----------
    effected_vocals_track : StrPath, optional
        The path to an effected vocals track.
    song_dir : StrPath, optional
        The path to a song directory.
    model_name : str, optional
        The name of a voice model.

    Returns
    -------
    str
        The song cover name

    """
    song_name = "Unknown"
    if song_dir and (song_path := _get_input_audio_path(song_dir)):
        song_name = song_path.stem.removeprefix("00_")
    model_name = model_name or _get_model_name(effected_vocals_track, song_dir)

    return f"{song_name} ({model_name} Ver)"


def _get_youtube_id(url: str, ignore_playlist: bool = True) -> str:
    """
    Get the id of a YouTube video or playlist.

    Parameters
    ----------
    url : str
        URL which points to a YouTube video or playlist.
    ignore_playlist : bool, default=True
        Whether to get the id of the first video in a playlist or the
        playlist id itself.

    Returns
    -------
    str
        The id of a YouTube video or playlist.

    Raises
    ------
    YoutubeUrlError
        If the provided URL does not point to a YouTube video
        or playlist.

    """
    yt_id = None
    validate_url(url)
    query = urlparse(url)
    if query.hostname == "youtu.be":
        yt_id = query.query[2:] if query.path[1:] == "watch" else query.path[1:]

    elif query.hostname in {"www.youtube.com", "youtube.com", "music.youtube.com"}:
        if not ignore_playlist:
            with suppress(KeyError):
                yt_id = parse_qs(query.query)["list"][0]
        elif query.path == "/watch":
            yt_id = parse_qs(query.query)["v"][0]
        elif query.path[:7] == "/watch/":
            yt_id = query.path.split("/")[1]
        elif query.path[:7] == "/embed/" or query.path[:3] == "/v/":
            yt_id = query.path.split("/")[2]
    if yt_id is None:
        raise YoutubeUrlError(url=url, playlist=True)

    return yt_id


def init_song_dir(
    source: str,
    progress_bar: gr.Progress | None = None,
    percentage: float = 0.5,
) -> tuple[Path, SongSourceType]:
    """
    Initialize a directory for a song provided by a given source.


    The song directory is initialized as follows:

    * If the source is a YouTube URL, the id of the video which
    that URL points to is extracted. A new song directory with the name
    of that id is then created, if it does not already exist.
    * If the source is a path to a local audio file, the hash of
    that audio file is extracted. A new song directory with the name of
    that hash is then created, if it does not already exist.
    * if the source is a path to an existing song directory, then
    that song directory is used as is.

    Parameters
    ----------
    source : str
        The source providing the song to initialize a directory for.
    progress_bar : gr.Progress, optional
        Gradio progress bar to update.
    percentage : float, default=0.5
        Percentage to display in the progress bar.

    Returns
    -------
    song_dir : Path
        The path to the initialized song directory.
    source_type : SongSourceType
        The type of source provided.

    Raises
    ------
    NotProvidedError
        If no source is provided.
    InvalidLocationError
        If a provided path points to a directory that is not located in
        the root of the intermediate audio base directory.
    NotFoundError
        If the provided source is a path to a file that does not exist.

    """
    if not source:
        raise NotProvidedError(entity=Entity.SOURCE, ui_msg=UIMessage.NO_AUDIO_SOURCE)
    source_path = Path(source)

    display_progress("[~] Initializing song directory...", percentage, progress_bar)

    # if source is a path to an existing song directory
    if source_path.is_dir():
        if source_path.parent != INTERMEDIATE_AUDIO_BASE_DIR:
            raise InvalidLocationError(
                entity=Entity.DIRECTORY,
                location=Location.INTERMEDIATE_AUDIO_ROOT,
                path=source_path,
            )
        display_progress(
            "[~] Using existing song directory...",
            percentage,
            progress_bar,
        )
        source_type = SongSourceType.SONG_DIR
        return source_path, source_type

    # if source is a URL
    if urlparse(source).scheme == "https":
        source_type = SongSourceType.URL
        song_id = _get_youtube_id(source)

    # if source is a path to a local audio file
    elif source_path.is_file():
        source_type = SongSourceType.FILE
        song_id = get_file_hash(source_path)
    else:
        raise NotFoundError(entity=Entity.FILE, location=source_path)

    song_dir_path = INTERMEDIATE_AUDIO_BASE_DIR / song_id

    song_dir_path.mkdir(parents=True, exist_ok=True)

    return song_dir_path, source_type


def _get_youtube_audio(url: str, directory: StrPath) -> Path:
    """
    Download audio from a YouTube video.

    Parameters
    ----------
    url : str
        URL which points to a YouTube video.
    directory : StrPath
        The directory to save the downloaded audio file to.

    Returns
    -------
    Path
        The path to the downloaded audio file.

    Raises
    ------
    YoutubeUrlError
        If the provided URL does not point to a YouTube video.

    """
    static_ffmpeg.add_paths()
    validate_url(url)
    outtmpl = str(Path(directory, "00_%(title)s"))
    ydl_opts = {
        "quiet": True,
        "no_warnings": True,
        "format": "bestaudio",
        "outtmpl": outtmpl,
        "ignoreerrors": True,
        "nocheckcertificate": True,
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "wav",
                "preferredquality": 0,
            },
        ],
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        result = ydl.extract_info(url, download=True)
        if not result:
            raise YoutubeUrlError(url, playlist=False)
        file = ydl.prepare_filename(result, outtmpl=f"{outtmpl}.wav")

    return Path(file)


def retrieve_song(
    source: str,
    progress_bar: gr.Progress | None = None,
    percentage: float = 0.5,
) -> tuple[Path, Path]:
    """
    Retrieve a song from a source that can either be a YouTube URL, a
    local audio file or a song directory.

    Parameters
    ----------
    source : str
        A Youtube URL, the path to a local audio file or the path to a
        song directory.
    progress_bar : gr.Progress, optional
        Gradio progress bar to update.
    percentage : float, default=0.5
        Percentage to display in the progress bar.

    Returns
    -------
    song : Path
        The path to the retrieved song.
    song_dir : Path
        The path to the song directory containing the retrieved song.

    Raises
    ------
    NotProvidedError
        If no source is provided.

    """
    if not source:
        raise NotProvidedError(entity=Entity.SOURCE, ui_msg=UIMessage.NO_AUDIO_SOURCE)

    song_dir_path, source_type = init_song_dir(source, progress_bar, percentage)
    song_path = _get_input_audio_path(song_dir_path)

    if not song_path:
        if source_type == SongSourceType.URL:
            display_progress("[~] Downloading song...", percentage, progress_bar)
            song_url = source.split("&")[0]
            song_path = _get_youtube_audio(song_url, song_dir_path)

        else:
            display_progress("[~] Copying song...", percentage, progress_bar)
            source_path = Path(source)
            song_name = f"00_{source_path.name}"
            song_path = song_dir_path / song_name
            shutil.copyfile(source_path, song_path)

    return song_path, song_dir_path


def separate_audio(
    audio_track: StrPath,
    song_dir: StrPath,
    model_name: SeparationModel,
    segment_size: int,
    display_msg: str = "[~] Separating audio...",
    progress_bar: gr.Progress | None = None,
    percentage: float = 0.5,
) -> tuple[Path, Path]:
    """
    Separate an audio track into a primary stem and a secondary stem.

    Parameters
    ----------
    audio_track : StrPath
        The path to the audio track to separate.
    song_dir : StrPath
        The path to the song directory where the separated primary stem
        and secondary stem will be saved.
    model_name : str
        The name of the model to use for audio separation.
    segment_size : int
        The segment size to use for audio separation.
    display_msg : str
        The message to display when separating the audio track.
    progress_bar : gr.Progress, optional
        Gradio progress bar to update.
    percentage : float, default=0.5
        Percentage to display in the progress bar.

    Returns
    -------
    primary_path : Path
        The path to the separated primary stem.
    secondary_path : Path
        The path to the separated secondary stem.

    """
    audio_path, song_dir_path = validate_all_exist(
        [(audio_track, Entity.AUDIO_TRACK), (song_dir, Entity.SONG_DIR)],
    )

    args_dict = SeparatedAudioMetaData(
        audio_track=FileMetaData(
            name=audio_path.name,
            hash_id=get_file_hash(audio_path),
        ),
        model_name=model_name,
        segment_size=segment_size,
    ).model_dump()

    paths = [
        get_unique_base_path(
            song_dir_path,
            prefix,
            args_dict,
        ).with_suffix(suffix)
        for prefix in ["11_Stem_Primary", "11_Stem_Secondary"]
        for suffix in [".wav", ".json"]
    ]

    (
        primary_path,
        primary_json_path,
        secondary_path,
        secondary_json_path,
    ) = paths

    if not all(path.exists() for path in paths):
        display_progress(display_msg, percentage, progress_bar)
        audio_separator = _get_audio_separator(
            output_dir=song_dir_path,
            segment_size=segment_size,
        )
        audio_separator.load_model(model_name)
        audio_separator.separate(
            str(audio_path),
            primary_output_name=primary_path.stem,
            secondary_output_name=secondary_path.stem,
        )
        json_dump(args_dict, primary_json_path)
        json_dump(args_dict, secondary_json_path)

    return primary_path, secondary_path


def _add_effects(
    audio_track: StrPath,
    output_file: StrPath,
    room_size: float = 0.15,
    wet_level: float = 0.2,
    dry_level: float = 0.8,
    damping: float = 0.7,
) -> None:
    """
    Add high-pass filter, compressor and reverb effects to an audio
    track.

    Parameters
    ----------
    audio_track : StrPath
        The path to the audio track to add effects to.
    output_file : StrPath
        The path to the file to save the effected audio track to.
    room_size : float, default=0.15
        The room size of the reverb effect.
    wet_level : float, default=0.2
        The wetness level of the reverb effect.
    dry_level : float, default=0.8
        The dryness level of the reverb effect.
    damping : float, default=0.7
        The damping of the reverb effect.

    """
    board = pedalboard_.Pedalboard(
        [
            pedalboard.HighpassFilter(),
            pedalboard.Compressor(ratio=4, threshold_db=-15),
            pedalboard.Reverb(
                room_size=room_size,
                dry_level=dry_level,
                wet_level=wet_level,
                damping=damping,
            ),
        ],
    )

    with (
        pedalboard_io.AudioFile(str(audio_track)) as f,
        pedalboard_io.AudioFile(
            str(output_file),
            "w",
            f.samplerate,
            f.num_channels,
        ) as o,
    ):
        # Read one second of audio at a time, until the file is empty:
        while f.tell() < f.frames:
            chunk = f.read(int(f.samplerate))
            effected = board(chunk, f.samplerate, reset=False)
            o.write(effected)


def postprocess(
    vocals_track: StrPath,
    song_dir: StrPath,
    room_size: float = 0.15,
    wet_level: float = 0.2,
    dry_level: float = 0.8,
    damping: float = 0.7,
    progress_bar: gr.Progress | None = None,
    percentage: float = 0.5,
) -> Path:
    """
    Apply high-pass filter, compressor and reverb effects to a vocals
    track.

    Parameters
    ----------
    vocals_track : StrPath
        The path to the vocals track to add effects to.
    song_dir : StrPath
        The path to the song directory where the effected vocals track
        will be saved.
    room_size : float, default=0.15
        The room size of the reverb effect.
    wet_level : float, default=0.2
        The wetness level of the reverb effect.
    dry_level : float, default=0.8
        The dryness level of the reverb effect.
    damping : float, default=0.7
        The damping of the reverb effect.
    progress_bar : gr.Progress, optional
        Gradio progress bar to update.
    percentage : float, default=0.5
        Percentage to display in the progress bar.

    Returns
    -------
    Path
        The path to the effected vocals track.

    """
    vocals_path, song_dir_path = validate_all_exist(
        [(vocals_track, Entity.VOCALS_TRACK), (song_dir, Entity.SONG_DIR)],
    )

    vocals_path = wavify(
        vocals_path,
        song_dir_path,
        "30_Input",
        accepted_formats={AudioExt.M4A, AudioExt.AAC},
        progress_bar=progress_bar,
        percentage=percentage,
    )

    args_dict = EffectedVocalsMetaData(
        vocals_track=FileMetaData(
            name=vocals_path.name,
            hash_id=get_file_hash(vocals_path),
        ),
        room_size=room_size,
        wet_level=wet_level,
        dry_level=dry_level,
        damping=damping,
    ).model_dump()

    paths = [
        get_unique_base_path(
            song_dir_path,
            "31_Vocals_Effected",
            args_dict,
        ).with_suffix(suffix)
        for suffix in [".wav", ".json"]
    ]

    effected_vocals_path, effected_vocals_json_path = paths

    if not all(path.exists() for path in paths):
        display_progress(
            "[~] Applying audio effects to vocals...",
            percentage,
            progress_bar,
        )
        _add_effects(
            vocals_path,
            effected_vocals_path,
            room_size,
            wet_level,
            dry_level,
            damping,
        )
        json_dump(args_dict, effected_vocals_json_path)
    return effected_vocals_path


def _pitch_shift(audio_track: StrPath, output_file: StrPath, n_semi_tones: int) -> None:
    """
    Pitch-shift an audio track.

    Parameters
    ----------
    audio_track : StrPath
        The path to the audio track to pitch-shift.
    output_file : StrPath
        The path to the file to save the pitch-shifted audio track to.
    n_semi_tones : int
        The number of semi-tones to pitch-shift the audio track by.

    """
    static_sox.add_paths()
    y, sr = sf.read(audio_track)
    tfm = sox.Transformer()
    tfm.pitch(n_semi_tones)
    y_shifted = tfm.build_array(input_array=y, sample_rate_in=sr)
    sf.write(output_file, y_shifted, sr)


def pitch_shift(
    audio_track: StrPath,
    song_dir: StrPath,
    n_semitones: int,
    display_msg: str = "[~] Pitch-shifting audio...",
    progress_bar: gr.Progress | None = None,
    percentage: float = 0.5,
) -> Path:
    """
    Pitch shift an audio track by a given number of semi-tones.

    Parameters
    ----------
    audio_track : StrPath
        The path to the audio track to pitch shift.
    song_dir : StrPath
        The path to the song directory where the pitch-shifted audio
        track will be saved.
    n_semitones : int
        The number of semi-tones to pitch-shift the audio track by.
    display_msg : str
        The message to display when pitch-shifting the audio track.
    progress_bar : gr.Progress, optional
        Gradio progress bar to update.
    percentage : float, default=0.5
        Percentage to display in the progress bar.

    Returns
    -------
    Path
        The path to the pitch-shifted audio track.

    """
    audio_path, song_dir_path = validate_all_exist(
        [(audio_track, Entity.AUDIO_TRACK), (song_dir, Entity.SONG_DIR)],
    )

    audio_path = wavify(
        audio_path,
        song_dir_path,
        "40_Input",
        accepted_formats={AudioExt.M4A, AudioExt.AAC},
        progress_bar=progress_bar,
        percentage=percentage,
    )

    shifted_audio_path = audio_path

    if n_semitones != 0:
        args_dict = PitchShiftMetaData(
            audio_track=FileMetaData(
                name=audio_path.name,
                hash_id=get_file_hash(audio_path),
            ),
            n_semitones=n_semitones,
        ).model_dump()

        paths = [
            get_unique_base_path(
                song_dir_path,
                "41_Audio_Shifted",
                args_dict,
            ).with_suffix(suffix)
            for suffix in [".wav", ".json"]
        ]

        shifted_audio_path, shifted_audio_json_path = paths

        if not all(path.exists() for path in paths):
            display_progress(display_msg, percentage, progress_bar)
            _pitch_shift(audio_path, shifted_audio_path, n_semitones)
            json_dump(args_dict, shifted_audio_json_path)

    return shifted_audio_path


def mix_song(
    audio_track_gain_pairs: Sequence[tuple[StrPath, int]],
    song_dir: StrPath,
    output_sr: int = 44100,
    output_format: AudioExt = AudioExt.MP3,
    output_name: str | None = None,
    progress_bar: gr.Progress | None = None,
    percentage: float = 0.5,
) -> Path:
    """
    Mix multiple audio tracks to create a song.

    Parameters
    ----------
    audio_track_gain_pairs : Sequence[tuple[StrPath, int]]
        A sequence of pairs each containing the path to an audio track
        and the gain to apply to it.
    song_dir : StrPath
        The path to the song directory where the song will be saved.
    output_sr : int, default=44100
        The sample rate of the mixed song.
    output_format : AudioExt, default=AudioExt.MP3
        The audio format of the mixed song.
    output_name : str, optional
        The name of the mixed song.
    progress_bar : gr.Progress, optional
        Gradio progress bar to update.
    percentage : float, default=0.5
        Percentage to display in the progress bar.

    Returns
    -------
    Path
        The path to the song cover.

    """
    mix_path = mix_audio(
        audio_track_gain_pairs,
        song_dir,
        output_sr,
        output_format,
        content_type=MixedAudioType.SONG,
        progress_bar=progress_bar,
        percentage=percentage,
    )
    output_name = output_name or get_song_cover_name(
        audio_track_gain_pairs[0][0],
        song_dir,
        None,
    )
    song_path = OUTPUT_AUDIO_DIR / f"{output_name}.{output_format}"
    return copy_file_safe(mix_path, song_path)


@app.command()
def run_pipeline(
    source: Annotated[str, Parameter(group=arguments_group)],
    model_name: Annotated[str, Parameter(group=arguments_group)],
    n_octaves: Annotated[int, Parameter(group=main_options_group)] = 0,
    n_semitones: Annotated[int, Parameter(group=main_options_group)] = 0,
    f0_methods: Annotated[
        Sequence[F0Method] | None,
        Parameter(
            group=voice_synthesis_options_group,
            show_default=True,
            show_choices=True,
        ),
    ] = None,
    index_rate: Annotated[
        float,
        Parameter(
            group=voice_synthesis_options_group,
            validator=validators.Number(gte=0, lte=1),
        ),
    ] = 0.5,
    filter_radius: Annotated[
        int,
        Parameter(
            group=voice_synthesis_options_group,
            validator=validators.Number(gte=0, lte=7),
        ),
    ] = 3,
    rms_mix_rate: Annotated[
        float,
        Parameter(
            group=voice_synthesis_options_group,
            validator=validators.Number(gte=0, lte=1),
        ),
    ] = 0.25,
    protect_rate: Annotated[
        float,
        Parameter(
            group=voice_synthesis_options_group,
            validator=validators.Number(gte=0, lte=0.5),
        ),
    ] = 0.33,
    hop_length: Annotated[
        int,
        Parameter(
            group=voice_synthesis_options_group,
            validator=validators.Number(gte=1, lte=512),
        ),
    ] = 128,
    split_vocals: Annotated[
        bool,
        Parameter(group=vocal_enrichment_options_group),
    ] = False,
    autotune_vocals: Annotated[
        bool,
        Parameter(group=vocal_enrichment_options_group, show_env_var=False),
    ] = False,
    autotune_strength: Annotated[
        float,
        Parameter(
            group=vocal_enrichment_options_group,
            validator=validators.Number(gte=0, lte=1),
        ),
    ] = 1.0,
    clean_vocals: Annotated[
        bool,
        Parameter(group=vocal_enrichment_options_group),
    ] = False,
    clean_strength: Annotated[
        float,
        Parameter(
            group=vocal_enrichment_options_group,
            validator=validators.Number(gte=0, lte=1),
        ),
    ] = 0.7,
    embedder_model: Annotated[
        EmbedderModel,
        Parameter(group=speaker_embeddings_options_group),
    ] = EmbedderModel.CONTENTVEC,
    embedder_model_custom: Annotated[
        StrPath | None,
        Parameter(
            group=speaker_embeddings_options_group,
            validator=validators.Path(exists=True, file_okay=False, dir_okay=True),
        ),
    ] = None,
    sid: Annotated[int, Parameter(group=speaker_embeddings_options_group)] = 0,
    room_size: Annotated[
        float,
        Parameter(
            group=vocal_post_processing_options_group,
            validator=validators.Number(gte=0, lte=1),
        ),
    ] = 0.15,
    wet_level: Annotated[
        float,
        Parameter(
            group=vocal_post_processing_options_group,
            validator=validators.Number(gte=0, lte=1),
        ),
    ] = 0.2,
    dry_level: Annotated[
        float,
        Parameter(
            group=vocal_post_processing_options_group,
            validator=validators.Number(gte=0, lte=1),
        ),
    ] = 0.8,
    damping: Annotated[
        float,
        Parameter(
            group=vocal_post_processing_options_group,
            validator=validators.Number(gte=0, lte=1),
        ),
    ] = 0.7,
    main_gain: Annotated[int, Parameter(group=audio_mixing_options_group)] = 0,
    inst_gain: Annotated[int, Parameter(group=audio_mixing_options_group)] = 0,
    backup_gain: Annotated[int, Parameter(group=audio_mixing_options_group)] = 0,
    output_sr: Annotated[int, Parameter(group=audio_mixing_options_group)] = 44100,
    output_format: Annotated[
        AudioExt,
        Parameter(group=audio_mixing_options_group),
    ] = AudioExt.MP3,
    output_name: Annotated[
        str | None,
        Parameter(group=audio_mixing_options_group),
    ] = None,
    progress_bar: Annotated[gr.Progress | None, Parameter(show=False)] = None,
    is_cli: Annotated[bool, Parameter(show=False)] = True,
) -> tuple[Path, ...]:
    """
    Run the song cover generation pipeline.

    Parameters
    ----------
    source : str
        A Youtube URL, the path to a local audio file or the path to a
        song directory.
    model_name : str
        The name of the voice model to use for vocal conversion.
    n_octaves : int, default=0
        The number of octaves to pitch-shift the converted vocals by.
        Use 1 for male-to-female and -1 for vice-versa.
    n_semitones : int, default=0
        The number of semi-tones to pitch-shift the converted vocals,
        instrumentals, and backup vocals by.Altering this slightly
        reduces sound quality
    f0_methods : Sequence[F0Method], optional
        The methods to use for pitch extraction during vocal
        conversion. If not provided, will default to the rmvpe method,
        which is generally recommended.
    index_rate : float, default=0.5
        The rate of influence of the index file. Increase to bias the
        vocal conversion towards the accent of the used voice model.
        Decrease to potentially reduce artifacts at the cost of accent
        accuracy.
    filter_radius : int, default=3
        A number which, if greater than 3, applies median filtering to
        the pitch values extracted during vocal conversion. Can help
        reduce breathiness in the converted vocals.
    rms_mix_rate : float, default=0.25
        Blending rate for the volume envelope of the vocals track.
        Controls how much to mimic the loudness of the input vocals (0)
        or a fixed loudness (1) during vocal conversion.
    protect_rate : float, default=0.33
        A coefficient which controls the extent to which consonants and
        breathing sounds are protected from artifacts during vocal
        conversion. A higher value offers more protection but may worsen
        the indexing effect.
    hop_length : int, default=128
        Controls how often the CREPE-based pitch extraction method
        checks for pitch changes during vocal conversion measured in
        milliseconds. Lower values lead to longer conversion times and a
        higher risk of voice, but better pitch accuracy.
    split_vocals : bool, default=False
        Whether to split the main vocals track into smaller segments
        before converting it. This can improve output quality for longer
        main vocal tracks.
    autotune_vocals : bool, default=False
        Whether to apply autotune to the converted vocals.
    autotune_strength : float, default=1.0
        The intensity of the autotune effect to apply to the converted
        vocals. Higher values result in stronger snapping to the
        chromatic grid.
    clean_vocals : bool, default=False
        Whether to apply noise reduction algorithms to the converted
        vocals.
    clean_strength : float, default=0.7
        The intensity of the cleaning to apply to the converted vocals.
        Higher values result in stronger cleaning, but may lead to a
        more compressed sound.
    embedder_model : EmbedderModel, default=EmbedderModel.CONTENTVEC
        The model to use for generating speaker embeddings during vocal
        conversion.
    embedder_model_custom : StrPath, optional
        The path to a directory with a custom model to use for
        generating speaker embeddings during vocal conversion. Only
        applicable if `embedder_model` is set to `custom`.
    sid : int, default=0
        The id of the speaker to use for multi-speaker RVC models.
    room_size : float, default=0.15
        The room size of the reverb effect to apply to the converted
        vocals. Increase for longer reverb time.
    wet_level : float, default=0.2
        The loudness of the converted vocals with reverb effect applied.
    dry_level : float, default=0.8
        The loudness of the converted vocals wihout reverb effect
        applied.
    damping : float, default=0.7
        The absorption of high frequencies in the reverb effect applied
        to the converted vocals.
    main_gain : int, default=0
        The gain to apply to the post-processed vocals. Measured in dB.
    inst_gain : int, default=0
        The gain to apply to the pitch-shifted instrumentals. Measured
        in dB.
    backup_gain : int, default=0
        The gain to apply to the pitch-shifted backup vocals. Measured
        in dB.
    output_sr : int, default=44100
        The sample rate of the song cover.
    output_format : AudioExt, default=AudioExt.MP3
        The audio format of the song cover.
    output_name : str, optional
        The name of the song cover.
    progress_bar : gr.Progress, optional
        Gradio progress bar to update.

    Returns
    -------
    tuple[Path,...]
        The path to the generated song cover and the paths to any
        intermediate audio files that were generated.

    """
    start_time = time.perf_counter()
    validate_exists(model_name, Entity.MODEL_NAME)
    display_progress("[~] Starting song cover generation pipeline...", 0, progress_bar)
    song, song_dir = retrieve_song(
        source,
        progress_bar=progress_bar,
        percentage=0 / 9,
    )
    vocals_track, instrumentals_track = separate_audio(
        song,
        song_dir,
        SeparationModel.UVR_MDX_NET_VOC_FT,
        SegmentSize.SEG_512,
        display_msg="[~] Separating vocals from instrumentals...",
        progress_bar=progress_bar,
        percentage=1 / 9,
    )
    backup_vocals_track, main_vocals_track = separate_audio(
        vocals_track,
        song_dir,
        SeparationModel.UVR_MDX_NET_KARA_2,
        SegmentSize.SEG_512,
        display_msg="[~] Separating main vocals from backup vocals...",
        progress_bar=progress_bar,
        percentage=2 / 9,
    )

    reverb_track, vocals_dereverb_track = separate_audio(
        main_vocals_track,
        song_dir,
        SeparationModel.REVERB_HQ_BY_FOXJOY,
        SegmentSize.SEG_256,
        display_msg="[~] De-reverbing vocals...",
        progress_bar=progress_bar,
        percentage=3 / 9,
    )
    converted_vocals_track = convert(
        audio_track=vocals_dereverb_track,
        directory=song_dir,
        model_name=model_name,
        n_octaves=n_octaves,
        n_semitones=n_semitones,
        f0_methods=f0_methods,
        index_rate=index_rate,
        filter_radius=filter_radius,
        rms_mix_rate=rms_mix_rate,
        protect_rate=protect_rate,
        hop_length=hop_length,
        split_audio=split_vocals,
        autotune_audio=autotune_vocals,
        autotune_strength=autotune_strength,
        clean_audio=clean_vocals,
        clean_strength=clean_strength,
        embedder_model=embedder_model,
        embedder_model_custom=embedder_model_custom,
        sid=sid,
        content_type=RVCContentType.VOCALS,
        progress_bar=progress_bar,
        percentage=4 / 9,
    )
    effected_vocals_track = postprocess(
        converted_vocals_track,
        song_dir,
        room_size,
        wet_level,
        dry_level,
        damping,
        progress_bar=progress_bar,
        percentage=5 / 9,
    )
    shifted_instrumentals_track = pitch_shift(
        instrumentals_track,
        song_dir,
        n_semitones,
        display_msg="[~] Pitch-shifting instrumentals...",
        progress_bar=progress_bar,
        percentage=6 / 9,
    )

    shifted_backup_vocals_track = pitch_shift(
        backup_vocals_track,
        song_dir,
        n_semitones,
        display_msg="[~] Pitch-shifting backup vocals...",
        progress_bar=progress_bar,
        percentage=7 / 9,
    )

    song_cover = mix_song(
        [
            (effected_vocals_track, main_gain),
            (shifted_instrumentals_track, inst_gain),
            (shifted_backup_vocals_track, backup_gain),
        ],
        song_dir,
        output_sr,
        output_format,
        output_name,
        progress_bar=progress_bar,
        percentage=8 / 9,
    )
    if is_cli:
        table = Table()
        table.add_column("Type")
        table.add_column("Path")
        for name, path in zip(
            [
                "Song",
                "Vocals",
                "Instrumentals",
                "Main vocals",
                "Backup vocals",
                "De-reverbed main vocals",
                "Main vocals reverb",
                "Converted vocals",
                "Post-processed vocals",
                "Pitch-shifted instrumentals",
                "Pitch-shifted backup vocals",
            ],
            [
                song,
                vocals_track,
                instrumentals_track,
                main_vocals_track,
                backup_vocals_track,
                vocals_dereverb_track,
                reverb_track,
                converted_vocals_track,
                effected_vocals_track,
                shifted_instrumentals_track,
                shifted_backup_vocals_track,
            ],
            strict=True,
        ):
            table.add_row(name, f"[green]{path}")
            rprint("[+] Song cover succesfully generated!")
            rprint()
            rprint("Elapsed time:", format_duration(time.perf_counter() - start_time))
            rprint(Panel(f"[green]{song_cover}", title="Song Cover Path"))
            rprint(Panel(table, title="Intermediate Audio Files"))
    return (
        song_cover,
        song,
        vocals_track,
        instrumentals_track,
        main_vocals_track,
        backup_vocals_track,
        vocals_dereverb_track,
        reverb_track,
        converted_vocals_track,
        effected_vocals_track,
        shifted_instrumentals_track,
        shifted_backup_vocals_track,
    )
