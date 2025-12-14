"""
Module which defines functions that faciliatate song cover generation
using RVC.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import lazy_loader as lazy

import logging
import operator
import shutil
from contextlib import suppress
from functools import cache
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from pydantic import ValidationError

from ultimate_rvc.common import NODE_PATH, SEPARATOR_MODELS_DIR
from ultimate_rvc.core.common import (
    INTERMEDIATE_AUDIO_BASE_DIR,
    OUTPUT_AUDIO_DIR,
    copy_file_safe,
    display_progress,
    get_file_hash,
    json_dump,
    json_load,
    validate_model,
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
    validate_audio_dir_exists,
    validate_audio_file_exists,
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
    from collections.abc import Sequence

    import yt_dlp

    import gradio as gr

    import pedalboard
    import soundfile as sf
    import static_ffmpeg
    import static_sox

    # NOTE the only reason this is imported here is so we can annotate
    # the return type of the _get_audio_separator function
    from audio_separator.separator import Separator

else:
    static_ffmpeg = lazy.load("static_ffmpeg")
    static_sox = lazy.load("static_sox")
    yt_dlp = lazy.load("yt_dlp")
    pedalboard = lazy.load("pedalboard")
    sf = lazy.load("soundfile")

logger = logging.getLogger(__name__)


@cache
def _get_audio_separator(
    output_dir: StrPath = INTERMEDIATE_AUDIO_BASE_DIR,
    output_format: str = AudioExt.WAV,
    segment_size: int = SegmentSize.SEG_256,
    sample_rate: int = 44100,
) -> Separator:
    static_ffmpeg.add_paths(weak=True)
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
        use_autocast=False,
        output_dir=output_dir,
        output_format=output_format,
        sample_rate=sample_rate,
        mdx_params={
            "hop_length": 1024,
            "segment_size": segment_size,
            "overlap": 0.25,
            "batch_size": 1,
            "enable_denoise": True,
        },
    )


def initialize_audio_separator() -> None:
    """
    Initialize the audio separator by downloading the models it
    uses.

    """
    audio_separator = _get_audio_separator()
    for i, separator_model in enumerate(SeparationModel):
        if not Path(SEPARATOR_MODELS_DIR / separator_model).is_file():
            display_progress(
                f"Downloading {separator_model}...",
                i / len(SeparationModel),
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


def init_song_dir(source: str) -> tuple[Path, SongSourceType]:
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

    # if source is a path to an existing song directory
    if source_path.is_dir():
        if source_path.parent != INTERMEDIATE_AUDIO_BASE_DIR:
            raise InvalidLocationError(
                entity=Entity.DIRECTORY,
                location=Location.INTERMEDIATE_AUDIO_ROOT,
                path=source_path,
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


def _get_youtube_audio(
    url: str,
    directory: StrPath,
    cookiefile: StrPath | None = None,
) -> Path:
    """
    Download audio from a YouTube video.

    Parameters
    ----------
    url : str
        URL which points to a YouTube video.
    directory : StrPath
        The directory to save the downloaded audio file to.
    cookiefile : StrPath
        The path to a file containing cookies to use when downloading
        audio from Youtube.

    Returns
    -------
    Path
        The path to the downloaded audio file.

    Raises
    ------
    YoutubeUrlError
        If the provided URL does not point to a YouTube video.

    """
    static_ffmpeg.add_paths(weak=True)
    validate_url(url)
    outtmpl = str(Path(directory, "00_%(title)s.%(ext)s"))
    ydl_opts = {
        "quiet": True,
        "format": "bestaudio/best",
        "cookiefile": cookiefile,
        "outtmpl": outtmpl,
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "wav",
                "preferredquality": 0,
            },
        ],
        "js_runtimes": {
            "node": {"path": str(NODE_PATH)},
        },
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        result = ydl.extract_info(url, download=True)
        if not result:
            raise YoutubeUrlError(url, playlist=False)
        file = ydl.prepare_filename(result)

    return Path(file).with_suffix(".wav")


def retrieve_song(source: str, cookiefile: StrPath | None = None) -> tuple[Path, Path]:
    """
    Retrieve a song from a source that can either be a YouTube URL, a
    local audio file or a song directory.

    Parameters
    ----------
    source : str
        A Youtube URL, the path to a local audio file or the path to a
        song directory.
    cookiefile: StrPath, optional
        The path to a file containing cookies to use when downloading
        audio from Youtube.

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

    song_dir_path, source_type = init_song_dir(source)
    song_path = _get_input_audio_path(song_dir_path)

    if not song_path:
        if source_type == SongSourceType.URL:
            song_url = source.split("&", maxsplit=1)[0]
            song_path = _get_youtube_audio(song_url, song_dir_path, cookiefile)

        else:
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

    Returns
    -------
    primary_path : Path
        The path to the separated primary stem.
    secondary_path : Path
        The path to the separated secondary stem.

    """
    audio_path = validate_audio_file_exists(audio_track, Entity.AUDIO_TRACK)
    song_dir_path = validate_audio_dir_exists(song_dir, Entity.SONG_DIR)

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
        audio_separator = _get_audio_separator(
            output_dir=song_dir_path,
            segment_size=segment_size,
        )
        audio_separator.load_model(model_name)
        audio_separator.separate(
            str(audio_path),
            custom_output_names={
                audio_separator.model_instance.primary_stem_name: str(
                    primary_path.stem
                ),
                audio_separator.model_instance.secondary_stem_name: str(
                    secondary_path.stem,
                ),
            },
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
    board = pedalboard.Pedalboard(
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
        pedalboard.io.AudioFile(str(audio_track)) as f,
        pedalboard.io.AudioFile(
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

    Returns
    -------
    Path
        The path to the effected vocals track.

    """
    vocals_path = validate_audio_file_exists(vocals_track, Entity.VOCALS_TRACK)
    song_dir_path = validate_audio_dir_exists(song_dir, Entity.SONG_DIR)

    vocals_path = wavify(
        vocals_path,
        song_dir_path,
        "30_Input",
        accepted_formats={AudioExt.M4A, AudioExt.AAC},
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
    static_sox.add_paths(weak=True)
    # NOTE The lazy_import function does not work with sox
    # so we import it here manually
    import sox  # noqa: PLC0415

    y, sr = sf.read(audio_track)
    tfm = sox.Transformer()
    tfm.pitch(n_semi_tones)
    y_shifted = tfm.build_array(input_array=y, sample_rate_in=sr)
    sf.write(output_file, y_shifted, sr)


def pitch_shift(audio_track: StrPath, song_dir: StrPath, n_semitones: int) -> Path:
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

    Returns
    -------
    Path
        The path to the pitch-shifted audio track.

    """
    audio_path = validate_audio_file_exists(audio_track, Entity.AUDIO_TRACK)
    song_dir_path = validate_audio_dir_exists(song_dir, Entity.SONG_DIR)

    audio_path = wavify(
        audio_path,
        song_dir_path,
        "40_Input",
        accepted_formats={AudioExt.M4A, AudioExt.AAC},
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
            _pitch_shift(audio_path, shifted_audio_path, n_semitones)
            json_dump(args_dict, shifted_audio_json_path)

    return shifted_audio_path


def mix_song(
    audio_track_gain_pairs: Sequence[tuple[StrPath, int]],
    song_dir: StrPath,
    output_sr: int = 44100,
    output_format: AudioExt = AudioExt.MP3,
    output_name: str | None = None,
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
    )
    output_name = output_name or get_song_cover_name(
        audio_track_gain_pairs[0][0],
        song_dir,
        None,
    )
    song_path = OUTPUT_AUDIO_DIR / f"{output_name}.{output_format}"
    return copy_file_safe(mix_path, song_path)


def run_pipeline(
    source: str,
    model_name: str,
    n_octaves: int = 0,
    n_semitones: int = 0,
    f0_method: F0Method = F0Method.RMVPE,
    index_rate: float = 0.3,
    rms_mix_rate: float = 1.0,
    protect_rate: float = 0.33,
    split_vocals: bool = False,
    autotune_vocals: bool = False,
    autotune_strength: float = 1.0,
    proposed_pitch: bool = False,
    proposed_pitch_threshold: float = 155.0,
    clean_vocals: bool = False,
    clean_strength: float = 0.7,
    embedder_model: EmbedderModel = EmbedderModel.CONTENTVEC,
    custom_embedder_model: str | None = None,
    sid: int = 0,
    room_size: float = 0.15,
    wet_level: float = 0.2,
    dry_level: float = 0.8,
    damping: float = 0.7,
    main_gain: int = 0,
    inst_gain: int = 0,
    backup_gain: int = 0,
    output_sr: int = 44100,
    output_format: AudioExt = AudioExt.MP3,
    output_name: str | None = None,
    cookiefile: StrPath | None = None,
    progress_bar: gr.Progress | None = None,
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
    n_semitones : int, default=0
        The number of semi-tones to pitch-shift the converted vocals,
        instrumentals, and backup vocals by.
    f0_method: F0Method, default=F0Method.RMVPE
        The method to use for pitch extraction during vocal
        conversion.
    index_rate : float, default=0.3
        The influence of the index file on the vocal conversion.
    rms_mix_rate : float, default=1.0
        The blending rate of the volume envelope of the converted
        vocals.
    protect_rate : float, default=0.33
        The protect rate for consonants and breathing sounds during
        vocal conversion.
    split_vocals : bool, default=False
        Whether to perform audio splitting before converting the main
        vocals.
    autotune_vocals : bool, default=False
        Whether to apply autotune to the converted vocals.
    autotune_strength : float, default=1.0
        The strength of the autotune to apply to the converted vocals.
    proposed_pitch: bool = False,
        Whether to adjust the pitch of the converted vocals so that it
        matches the range of the voice model used.
    proposed_pitch_threshold: float = 155.0,
        The threshold for proposed pitch correction.
    clean_vocals : bool, default=False
        Whether to clean the converted vocals.
    clean_strength : float, default=0.7
        The intensity of the cleaning to apply to the converted vocals.
    embedder_model : EmbedderModel, default=EmbedderModel.CONTENTVEC
        The model to use for generating speaker embeddings during vocal
        conversion.
    custom_embedder_model : StrPath, optional
        The name of a custom embedder model to use for generating
        speaker embeddings during vocal conversion.
    sid : int, default=0
        The speaker id to use for multi-speaker models during vocal
        conversion.
    room_size : float, default=0.15
        The room size of the reverb effect to apply to the converted
        vocals.
    wet_level : float, default=0.2
        The wetness level of the reverb effect to apply to the converted
        vocals.
    dry_level : float, default=0.8
        The dryness level of the reverb effect to apply to the converted
        vocals.
    damping : float, default=0.7
        The damping of the reverb effect to apply to the converted
        vocals.
    main_gain : int, default=0
        The gain to apply to the post-processed vocals.
    inst_gain : int, default=0
        The gain to apply to the pitch-shifted instrumentals.
    backup_gain : int, default=0
        The gain to apply to the pitch-shifted backup vocals.
    output_sr : int, default=44100
        The sample rate of the song cover.
    output_format : AudioExt, default=AudioExt.MP3
        The audio format of the song cover.
    output_name : str, optional
        The name of the song cover.
    cookiefile : StrPath, optional
        The path to a file containing cookies to use when downloading
        audio from Youtube.
    progress_bar : gr.Progress, optional
        Gradio progress bar to update.

    Returns
    -------
    tuple[Path,...]
        The path to the generated song cover and the paths to any
        intermediate audio files that were generated.

    """
    validate_model(model_name, Entity.VOICE_MODEL)
    if embedder_model == EmbedderModel.CUSTOM:
        validate_model(custom_embedder_model, Entity.CUSTOM_EMBEDDER_MODEL)
    display_progress("[~] Retrieving song...", 0 / 9, progress_bar)
    song, song_dir = retrieve_song(source, cookiefile=cookiefile)
    display_progress("[~] Separating vocals from instrumentals...", 1 / 9, progress_bar)
    vocals_track, instrumentals_track = separate_audio(
        song,
        song_dir,
        SeparationModel.UVR_MDX_NET_VOC_FT,
        SegmentSize.SEG_512,
    )
    display_progress(
        "[~] Separating main vocals from backup vocals...",
        2 / 9,
        progress_bar,
    )
    backup_vocals_track, main_vocals_track = separate_audio(
        vocals_track,
        song_dir,
        SeparationModel.UVR_MDX_NET_KARA_2,
        SegmentSize.SEG_512,
    )
    display_progress("[~] De-reverbing vocals...", 3 / 9, progress_bar)
    reverb_track, vocals_dereverb_track = separate_audio(
        main_vocals_track,
        song_dir,
        SeparationModel.REVERB_HQ_BY_FOXJOY,
        SegmentSize.SEG_256,
    )
    display_progress("[~] Converting vocals...", 4 / 9, progress_bar)
    converted_vocals_track = convert(
        audio_track=vocals_dereverb_track,
        directory=song_dir,
        model_name=model_name,
        n_octaves=n_octaves,
        n_semitones=n_semitones,
        f0_method=f0_method,
        index_rate=index_rate,
        rms_mix_rate=rms_mix_rate,
        protect_rate=protect_rate,
        split_audio=split_vocals,
        autotune_audio=autotune_vocals,
        autotune_strength=autotune_strength,
        proposed_pitch=proposed_pitch,
        proposed_pitch_threshold=proposed_pitch_threshold,
        clean_audio=clean_vocals,
        clean_strength=clean_strength,
        embedder_model=embedder_model,
        custom_embedder_model=custom_embedder_model,
        sid=sid,
        content_type=RVCContentType.VOCALS,
    )
    display_progress("[~] Post-processing vocals...", 5 / 9, progress_bar)
    effected_vocals_track = postprocess(
        converted_vocals_track,
        song_dir,
        room_size,
        wet_level,
        dry_level,
        damping,
    )
    display_progress("[~] Pitch-shifting instrumentals...", 6 / 9, progress_bar)
    shifted_instrumentals_track = pitch_shift(
        instrumentals_track,
        song_dir,
        n_semitones,
    )
    display_progress("[~] Pitch-shifting backup vocals...", 7 / 9, progress_bar)
    shifted_backup_vocals_track = pitch_shift(
        backup_vocals_track,
        song_dir,
        n_semitones,
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
    )
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
