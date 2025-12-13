"""Module which defines functions to manage voice models."""

from __future__ import annotations

from typing import TYPE_CHECKING

import lazy_loader as lazy

import re
import shutil
import urllib.request
import zipfile
from pathlib import Path

from ultimate_rvc.common import (
    CUSTOM_EMBEDDER_MODELS_DIR,
    CUSTOM_PRETRAINED_MODELS_DIR,
    TRAINING_MODELS_DIR,
    VOICE_MODELS_DIR,
)
from ultimate_rvc.core.common import (
    copy_files_to_new_dir,
    display_progress,
    get_file_size,
    json_load,
    validate_model,
    validate_url,
)
from ultimate_rvc.core.exceptions import (
    Entity,
    Location,
    ModelEntity,
    ModelNotFoundError,
    NotFoundError,
    NotProvidedError,
    PretrainedModelExistsError,
    PretrainedModelNotAvailableError,
    UIMessage,
    UploadLimitError,
    UploadTypeError,
)
from ultimate_rvc.core.manage.common import delete_directory, get_items
from ultimate_rvc.core.manage.typing_extra import (
    PretrainedModelMetaDataTable,
    VoiceModelMetaData,
    VoiceModelMetaDataList,
    VoiceModelMetaDataPredicate,
    VoiceModelMetaDataTable,
    VoiceModelTagName,
)

if TYPE_CHECKING:
    from typing import NoReturn

    from _collections_abc import Sequence

    import requests

    import tqdm

    from ultimate_rvc.typing_extra import StrPath, TrainingSampleRate
else:
    requests = lazy.load("requests")
    tqdm = lazy.load("tqdm")

PUBLIC_MODELS_JSON = json_load(Path(__file__).parent / "public_models.json")
PUBLIC_MODELS_TABLE = VoiceModelMetaDataTable.model_validate(PUBLIC_MODELS_JSON)
PRETRAINED_MODELS_JSON = json_load(Path(__file__).parent / "custom_pretrains.json")
PRETRAINED_MODELS_TABLE = PretrainedModelMetaDataTable.model_validate(
    PRETRAINED_MODELS_JSON
)


def get_voice_model_names() -> list[str]:
    """
    Get the names of all saved voice models.

    Returns
    -------
    list[str]
        A list of names of all saved voice models.

    """
    return get_items(VOICE_MODELS_DIR)


def get_custom_embedder_model_names() -> list[str]:
    """
    Get the names of all saved custom embedder models.

    Returns
    -------
    list[str]
        A list of the names of all saved custom embedder models.

    """
    return get_items(CUSTOM_EMBEDDER_MODELS_DIR)


def get_custom_pretrained_model_names() -> list[str]:
    """
    Get the names of all saved custom pretrained models.

    Returns
    -------
    list[str]
        A list of the names of all saved custom pretrained models.

    """
    return get_items(CUSTOM_PRETRAINED_MODELS_DIR)


def get_training_model_names() -> list[str]:
    """
    Get the names of all saved training models.

    Returns
    -------
    list[str]
        A list of the names of all saved training models.

    """
    return get_items(TRAINING_MODELS_DIR)


def load_public_models_table(
    predicates: Sequence[VoiceModelMetaDataPredicate],
) -> VoiceModelMetaDataList:
    """
    Load table containing metadata of public voice models, optionally
    filtered by a set of predicates.

    Parameters
    ----------
    predicates : Sequence[VoiceModelMetaDataPredicate]
        Predicates to filter the metadata table by.

    Returns
    -------
    VoiceModelMetaDataList
        List containing metadata for each public voice model that
        satisfies the given predicates.

    """
    return [
        [
            model.name,
            model.description,
            model.tags,
            model.credit,
            model.added,
            model.url,
        ]
        for model in PUBLIC_MODELS_TABLE.models
        if all(predicate(model) for predicate in predicates)
    ]


def get_public_model_tags() -> list[VoiceModelTagName]:
    """
    Get the names of all valid public voice model tags.

    Returns
    -------
    list[str]
        A list of names of all valid public voice model tags.

    """
    return [tag.name for tag in PUBLIC_MODELS_TABLE.tags]


def filter_public_models_table(
    tags: Sequence[str],
    query: str,
) -> VoiceModelMetaDataList:
    """
    Filter table containing metadata of public voice models by tags and
    a search query.


    The search query is matched against the name, description, tags,
    credit,and added date of each entry in the metadata table. Case
    insensitive search is performed. If the search query is empty, the
    metadata table is filtered only bythe given tags.

    Parameters
    ----------
    tags : Sequence[str]
        Tags to filter the metadata table by.
    query : str
        Search query to filter the metadata table by.

    Returns
    -------
    VoiceModelMetaDataList
        List containing metadata for each public voice model that
        match the given tags and search query.

    """

    def _tags_predicate(model: VoiceModelMetaData) -> bool:
        return all(tag in model.tags for tag in tags)

    def _query_predicate(model: VoiceModelMetaData) -> bool:
        return (
            query.lower()
            in (
                f"{model.name} {model.description} {' '.join(model.tags)} "
                f"{model.credit} {model.added}"
            ).lower()
            if query
            else True
        )

    filter_fns = [_tags_predicate, _query_predicate]

    return load_public_models_table(filter_fns)


def _extract_voice_model(
    zip_file: StrPath,
    extraction_dir: StrPath,
    remove_incomplete: bool = True,
    remove_zip: bool = False,
) -> None:
    """
    Extract a zipped voice model to a directory.

    Parameters
    ----------
    zip_file : StrPath
        The path to a zip file containing the voice model to extract.
    extraction_dir : StrPath
        The path to the directory to extract the voice model to.

    remove_incomplete : bool, default=True
        Whether to remove the extraction directory if the extraction
        process fails.
    remove_zip : bool, default=False
        Whether to remove the zip file once the extraction process is
        complete.

    Raises
    ------
    NotFoundError
        If no model file is found in the extracted zip file.

    """
    extraction_path = Path(extraction_dir)
    zip_path = Path(zip_file)
    extraction_completed = False
    try:
        extraction_path.mkdir(parents=True)
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(extraction_path)
        file_path_map = {
            ext: Path(root, name)
            for root, _, files in extraction_path.walk()
            for name in files
            for ext in [".index", ".pth"]
            if Path(name).suffix == ext
            and Path(root, name).stat().st_size
            > 1024 * (100 if ext == ".index" else 1024 * 40)
        }
        if ".pth" not in file_path_map:
            raise NotFoundError(
                entity=Entity.MODEL_FILE,
                location=Location.EXTRACTED_ZIP_FILE,
                is_path=False,
            )

        # move model and index file to root of the extraction directory
        for file_path in file_path_map.values():
            file_path.rename(extraction_path / file_path.name)

        # remove any sub-directories within the extraction directory
        for path in extraction_path.iterdir():
            if path.is_dir():
                shutil.rmtree(path)
        extraction_completed = True
    finally:
        if not extraction_completed and remove_incomplete and extraction_path.is_dir():
            shutil.rmtree(extraction_path)
        if remove_zip and zip_path.exists():
            zip_path.unlink()


def download_voice_model(url: str, name: str) -> None:
    """
    Download a zipped voice model.

    Parameters
    ----------
    url : str
        An URL pointing to a location where the zipped voice model can
        be downloaded from.
    name : str
        The name to give to the downloaded voice model.

    """
    validate_url(url)
    extraction_path = validate_model(name, Entity.VOICE_MODEL, mode="not_exists")

    zip_name = url.rsplit("/", maxsplit=1)[-1].split("?", maxsplit=1)[0]

    # NOTE in case huggingface link is a direct link rather
    # than a resolve link then convert it to a resolve link
    url = re.sub(
        r"https://huggingface.co/([^/]+)/([^/]+)/blob/(.*)",
        r"https://huggingface.co/\1/\2/resolve/\3",
        url,
    )
    if "pixeldrain.com" in url:
        url = f"https://pixeldrain.com/api/file/{zip_name}"

    display_progress("[~] Downloading voice model ...")
    urllib.request.urlretrieve(url, zip_name)  # noqa: S310

    display_progress("[~] Extracting zip file...")
    _extract_voice_model(zip_name, extraction_path, remove_zip=True)


def _download_pretrained_model_file(
    url: str,
    destination: StrPath,
    progress_bar: tqdm.tqdm[NoReturn] | None = None,
) -> None:
    """
    Download a pretrained model file.

    Parameters
    ----------
    url : str
        The URL of the pretrained model file to download.
    destination : strPath
        The destination to save the downloaded pretrained model file to.
    progress_bar : tqdm.tqdm, optional
        TQDM progress bar to update.

    """
    response = requests.get(url, stream=True, timeout=10)
    response.raise_for_status()
    block_size = 1024
    destination_path = Path(destination)
    with destination_path.open("wb") as file:
        for data in response.iter_content(block_size):
            file.write(data)
            if progress_bar:
                progress_bar.update(len(data))


def download_pretrained_model(name: str, sample_rate: TrainingSampleRate) -> None:
    """
    Download a pretrained model.

    Parameters
    ----------
    name : str
        The name of the pretrained model to download.
    sample_rate : TrainingSampleRate
        The sample rate of the pretrained model to download.

    Raises
    ------
    PretrainedModelExistsError
        If a pretrained model with the provided name and sample rate
        already exists.
    PretrainedModelNotAvailableError
        If a pretrained model with the provided name and sample rate is
        not available for download.

    """
    CUSTOM_PRETRAINED_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = CUSTOM_PRETRAINED_MODELS_DIR / f"{name} {sample_rate}"
    if model_path.is_dir():
        raise PretrainedModelExistsError(name, sample_rate)

    if name not in PRETRAINED_MODELS_TABLE.names:
        raise PretrainedModelNotAvailableError(name)
    if sample_rate not in PRETRAINED_MODELS_TABLE.get_sample_rates(name):
        raise PretrainedModelNotAvailableError(name, sample_rate)
    paths = PRETRAINED_MODELS_TABLE[name][sample_rate]

    d_url = f"https://huggingface.co/{paths.D}"
    g_url = f"https://huggingface.co/{paths.G}"

    total_size = get_file_size(d_url) + get_file_size(g_url)

    model_path.mkdir(parents=True)
    import concurrent.futures  # noqa: PLC0415

    with (
        tqdm.tqdm(
            total=total_size,
            unit="iB",
            unit_scale=True,
            desc="Downloading files",
        ) as tqdm_bar,
        concurrent.futures.ThreadPoolExecutor() as executor,
    ):
        futures = [
            executor.submit(
                _download_pretrained_model_file,
                d_url,
                model_path / Path(paths.D).name,
                tqdm_bar,
            ),
            executor.submit(
                _download_pretrained_model_file,
                g_url,
                model_path / Path(paths.G).name,
                tqdm_bar,
            ),
        ]

        for future in futures:
            try:
                future.result()
            except requests.exceptions.RequestException as e:
                shutil.rmtree(model_path)
                raise PretrainedModelNotAvailableError(name, sample_rate) from e


def upload_voice_model(files: Sequence[StrPath], name: str) -> None:
    """
    Upload a voice model from either a zip file or a .pth file and an
    optional index file.

    Parameters
    ----------
    files : Sequence[StrPath]
        Paths to the files to upload.
    name : str
        The name to give to the uploaded voice model.

    Raises
    ------
    NotProvidedError
        If no file paths or name are provided.
    UploadTypeError
        If a single uploaded file is not a .pth file or a .zip file.
        If two uploaded files are not a .pth file and an .index file.
    UploadLimitError
        If more than two file paths are provided.

    """
    if not files:
        raise NotProvidedError(entity=Entity.FILES, ui_msg=UIMessage.NO_UPLOADED_FILES)
    model_dir_path = validate_model(name, Entity.VOICE_MODEL, mode="not_exists")
    sorted_file_paths = sorted([Path(f) for f in files], key=lambda f: f.suffix)
    match sorted_file_paths:
        case [file_path]:
            if file_path.suffix == ".pth":
                copy_files_to_new_dir([file_path], model_dir_path)
            # NOTE a .pth file is actually itself a zip file
            elif zipfile.is_zipfile(file_path):
                _extract_voice_model(file_path, model_dir_path)
            else:
                raise UploadTypeError(
                    entity=Entity.FILES,
                    valid_types=[".pth", ".zip"],
                    type_class="formats",
                    multiple=False,
                )
        case [index_path, pth_path]:
            if index_path.suffix == ".index" and pth_path.suffix == ".pth":
                copy_files_to_new_dir([index_path, pth_path], model_dir_path)
            else:
                raise UploadTypeError(
                    entity=Entity.FILES,
                    valid_types=[".pth", ".index"],
                    type_class="formats",
                    multiple=True,
                )
        case _:
            raise UploadLimitError(entity=Entity.FILES, limit="two")


def _extract_custom_embedder_model(
    zip_file: StrPath,
    extraction_dir: StrPath,
    remove_incomplete: bool = True,
    remove_zip: bool = False,
) -> None:
    """
    Extract a zipped custom embedder model to a directory.

    Parameters
    ----------
    zip_file : StrPath
        The path to a zip file containing the custom embedder model to
        extract.
    extraction_dir : StrPath
        The path to the directory to extract the custom embedder model
        to.

    remove_incomplete : bool, default=True
        Whether to remove the extraction directory if the extraction
        process fails.
    remove_zip : bool, default=False
        Whether to remove the zip file once the extraction process is
        complete.

    Raises
    ------
    NotFoundError
        If no pytorch_model.bin file or config.json file is found in
        the extracted zip file.

    """
    extraction_path = Path(extraction_dir)
    zip_path = Path(zip_file)
    extraction_completed = False
    try:
        extraction_path.mkdir(parents=True)
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(extraction_path)
        file_path_map = {
            file: Path(root, file)
            for root, _, files in extraction_path.walk()
            for file in files
            if file in {"pytorch_model.bin", "config.json"}
        }
        if "config.json" not in file_path_map:
            raise NotFoundError(
                entity=Entity.CONFIG_JSON_FILE,
                location=Location.EXTRACTED_ZIP_FILE,
                is_path=False,
            )
        if "pytorch_model.bin" not in file_path_map:
            raise NotFoundError(
                entity=Entity.MODEL_BIN_FILE,
                location=Location.EXTRACTED_ZIP_FILE,
                is_path=False,
            )

        # move pytorch_model.bin file and config.json file to root of
        # the extraction directory
        for file_path in file_path_map.values():
            file_path.rename(extraction_path / file_path.name)

        # remove any sub-directories within the extraction directory
        for path in extraction_path.iterdir():
            if path.is_dir():
                shutil.rmtree(path)
        extraction_completed = True
    finally:
        if not extraction_completed and remove_incomplete and extraction_path.is_dir():
            shutil.rmtree(extraction_path)
        if remove_zip and zip_path.exists():
            zip_path.unlink()


def upload_custom_embedder_model(files: Sequence[StrPath], name: str) -> None:
    """
    Upload a custom embedder model from either a zip file or a pair
    consisting of a pytorch_model.bin file and a config.json file.

    Parameters
    ----------
    files : Sequence[StrPath]
        Paths to the files to upload.
    name : str
        The name to give to the uploaded custom embedder model.

    Raises
    ------
    NotProvidedError
        If no name or file paths are provided.
    UploadTypeError
        If a single uploaded file is not a .zip file or two uploaded
        files are not named "pytorch_model.bin" and "config.json".
    UploadLimitError
        If more than two file paths are provided.

    """
    if not files:
        raise NotProvidedError(entity=Entity.FILES, ui_msg=UIMessage.NO_UPLOADED_FILES)
    model_dir_path = validate_model(
        name,
        Entity.CUSTOM_EMBEDDER_MODEL,
        mode="not_exists",
    )
    sorted_file_paths = sorted([Path(f) for f in files], key=lambda f: f.suffix)
    match sorted_file_paths:
        case [file_path]:
            if zipfile.is_zipfile(file_path):
                _extract_custom_embedder_model(file_path, model_dir_path)
            else:
                raise UploadTypeError(
                    entity=Entity.FILES,
                    valid_types=[".zip"],
                    type_class="formats",
                    multiple=False,
                )
        case [bin_path, json_path]:
            if bin_path.name == "pytorch_model.bin" and json_path.name == "config.json":
                copy_files_to_new_dir([bin_path, json_path], model_dir_path)
            else:
                raise UploadTypeError(
                    entity=Entity.FILES,
                    valid_types=["pytorch_model.bin", "config.json"],
                    type_class="names",
                    multiple=True,
                )
        case _:
            raise UploadLimitError(entity=Entity.FILES, limit="two")


def delete_models(
    directory: StrPath,
    names: Sequence[str],
    entity: ModelEntity = Entity.MODEL,
    ui_msg: UIMessage = UIMessage.NO_MODELS,
) -> None:
    """
    Delete the models with the provided names.

    Parameters
    ----------
    directory : StrPath
        The path to the directory containing the models to delete.
    names : Sequence[str]
        Names of the models to delete.
    entity : ModelEntity, optional
        The model entity being deleted.
    ui_msg : UIMessage, optional
        The message to display if no model names are provided.

    Raises
    ------
    NotProvidedError
        If no names of items are provided.
    ModelNotFoundError
        if an item with a provided name is not found.

    """
    if not names:
        raise NotProvidedError(entity=Entity.MODEL_NAMES, ui_msg=ui_msg)
    model_dir_paths: list[Path] = []
    for name in names:
        model_dir_path = Path(directory) / name
        if not model_dir_path.is_dir():
            raise ModelNotFoundError(entity=entity, name=name)
        model_dir_paths.append(model_dir_path)
    for model_dir_path in model_dir_paths:
        shutil.rmtree(model_dir_path)


def delete_voice_models(names: Sequence[str]) -> None:
    """
    Delete one or more voice models.

    Parameters
    ----------
    names : Sequence[str]
        Names of the voice models to delete.

    """
    delete_models(
        VOICE_MODELS_DIR,
        names,
        entity=Entity.VOICE_MODEL,
        ui_msg=UIMessage.NO_VOICE_MODELS,
    )


def delete_custom_embedder_models(names: Sequence[str]) -> None:
    """
    Delete one or more custom embedder models.

    Parameters
    ----------
    names : Sequence[str]
        Names of the custom embedder models to delete.

    """
    delete_models(
        CUSTOM_EMBEDDER_MODELS_DIR,
        names,
        entity=Entity.CUSTOM_EMBEDDER_MODEL,
        ui_msg=UIMessage.NO_CUSTOM_EMBEDDER_MODELS,
    )


def delete_custom_pretrained_models(names: Sequence[str]) -> None:
    """
    Delete one or more custom pretrained models.

    Parameters
    ----------
    names : Sequence[str]
        Names of the custom pretrained models to delete.

    """
    delete_models(
        CUSTOM_PRETRAINED_MODELS_DIR,
        names,
        entity=Entity.CUSTOM_PRETRAINED_MODEL,
        ui_msg=UIMessage.NO_CUSTOM_PRETRAINED_MODELS,
    )


def delete_training_models(names: Sequence[str]) -> None:
    """
    Delete one or more training models.

    Parameters
    ----------
    names : Sequence[str]
        Names of the training models to delete.

    """
    delete_models(
        TRAINING_MODELS_DIR,
        names,
        entity=Entity.TRAINING_MODEL,
        ui_msg=UIMessage.NO_TRAINING_MODELS,
    )


def delete_all_voice_models() -> None:
    """Delete all voice models."""
    delete_directory(VOICE_MODELS_DIR)


def delete_all_custom_embedder_models() -> None:
    """Delete all custom embedder models."""
    delete_directory(CUSTOM_EMBEDDER_MODELS_DIR)


def delete_all_custom_pretrained_models() -> None:
    """Delete all custom pretrained models."""
    delete_directory(CUSTOM_PRETRAINED_MODELS_DIR)


def delete_all_training_models() -> None:
    """Delete all training models."""
    delete_directory(TRAINING_MODELS_DIR)


def delete_all_models() -> None:
    """Delete all voice and training models."""
    display_progress("[~] Deleting all voice models ...", 0.0)
    delete_all_voice_models()
    display_progress("[~] Deleting all custom embedder models ...", 0.25)
    delete_all_custom_embedder_models()
    display_progress("[~] Deleting all custom pretrained models ...", 0.5)
    delete_all_custom_pretrained_models()
    display_progress("[~] Deleting all training models ...", 0.75)
    delete_all_training_models()
