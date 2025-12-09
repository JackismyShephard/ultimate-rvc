from typing import TYPE_CHECKING

import lazy_loader as lazy

import os
import pathlib
from concurrent.futures import ThreadPoolExecutor

from ultimate_rvc.common import (
    EMBEDDER_MODELS_DIR,
    PRETRAINED_MODELS_DIR,
    RVC_MODELS_DIR,
)

if TYPE_CHECKING:
    import requests

    import tqdm

else:
    tqdm = lazy.load("tqdm")
    requests = lazy.load("requests")


url_base = "https://huggingface.co/JackismyShephard/ultimate-rvc/resolve/main/Resources"

pretraineds_hifigan_list = [
    (
        "pretrained_v2/",
        [
            "f0D32k.pth",
            "f0D40k.pth",
            "f0D48k.pth",
            "f0G32k.pth",
            "f0G40k.pth",
            "f0G48k.pth",
        ],
    ),
]
pretraineds_refinegan_list = [("refinegan/", ["f0D32k.pth", "f0G32k.pth"])]
models_list = [("predictors/", ["rmvpe.pt", "fcpe.pt"])]
embedders_list = [
    ("embedders/contentvec/", ["pytorch_model.bin", "config.json"]),
    ("embedders/chinese_hubert_base/", ["pytorch_model.bin", "config.json"]),
    ("embedders/japanese_hubert_base/", ["pytorch_model.bin", "config.json"]),
    ("embedders/korean_hubert_base/", ["pytorch_model.bin", "config.json"]),
    ("embedders/spin/", ["pytorch_model.bin", "config.json"]),
    ("embedders/spin-v2/", ["pytorch_model.bin", "config.json"]),
]
executables_list = [
    ("", ["ffmpeg.exe", "ffprobe.exe"]),
]

folder_mapping_list = {
    "pretrained_v2/": str(PRETRAINED_MODELS_DIR / "hifi-gan/"),
    "refinegan/": str(PRETRAINED_MODELS_DIR / "refinegan/"),
    "embedders/contentvec/": str(EMBEDDER_MODELS_DIR / "contentvec/"),
    "embedders/chinese_hubert_base/": str(
        EMBEDDER_MODELS_DIR / "chinese_hubert_base/",
    ),
    "embedders/japanese_hubert_base/": str(
        EMBEDDER_MODELS_DIR / "japanese_hubert_base/",
    ),
    "embedders/korean_hubert_base/": str(
        EMBEDDER_MODELS_DIR / "korean_hubert_base/",
    ),
    "embedders/spin/": str(EMBEDDER_MODELS_DIR / "spin/"),
    "embedders/spin-v2/": str(EMBEDDER_MODELS_DIR / "spin-v2/"),
    "predictors/": str(RVC_MODELS_DIR / "predictors/"),
    "formant/": str(RVC_MODELS_DIR / "formant/"),
}


def get_file_size_if_missing(file_list):
    """
    Calculate the total size of files to be downloaded only if they do not exist locally.
    """
    total_size = 0
    for remote_folder, files in file_list:
        local_folder = folder_mapping_list.get(remote_folder, "")
        for file in files:
            destination_path = os.path.join(local_folder, file)
            if not pathlib.Path(destination_path).exists():
                url = f"{url_base}/{remote_folder}{file}"
                response = requests.head(url)
                total_size += int(response.headers.get("content-length", 0))
    return total_size


def download_file(url, destination_path, global_bar):
    """
    Download a file from the given URL to the specified destination path,
    updating the global progress bar as data is downloaded.
    """
    dir_name = os.path.dirname(destination_path)
    if dir_name:
        pathlib.Path(dir_name).mkdir(exist_ok=True, parents=True)
    response = requests.get(url, stream=True)
    block_size = 1024
    with pathlib.Path(destination_path).open("wb") as file:
        for data in response.iter_content(block_size):
            file.write(data)
            global_bar.update(len(data))


def download_mapping_files(file_mapping_list, global_bar):
    """
    Download all files in the provided file mapping list using a thread pool executor,
    and update the global progress bar as downloads progress.
    """
    with ThreadPoolExecutor() as executor:
        futures = []
        for remote_folder, file_list in file_mapping_list:
            local_folder = folder_mapping_list.get(remote_folder, "")
            for file in file_list:
                destination_path = os.path.join(local_folder, file)
                if not pathlib.Path(destination_path).exists():
                    url = f"{url_base}/{remote_folder}{file}"
                    futures.append(
                        executor.submit(
                            download_file,
                            url,
                            destination_path,
                            global_bar,
                        ),
                    )
        for future in futures:
            future.result()


def split_pretraineds(pretrained_list):
    f0_list = []
    non_f0_list = []
    for folder, files in pretrained_list:
        f0_files = [f for f in files if f.startswith("f0")]
        non_f0_files = [f for f in files if not f.startswith("f0")]
        if f0_files:
            f0_list.append((folder, f0_files))
        if non_f0_files:
            non_f0_list.append((folder, non_f0_files))
    return f0_list, non_f0_list


pretraineds_hifigan_list, _ = split_pretraineds(pretraineds_hifigan_list)


def calculate_total_size(
    pretraineds_hifigan,
    models,
    exe,
):
    """
    Calculate the total size of all files to be downloaded based on selected categories.
    """
    total_size = 0
    if models:
        total_size += get_file_size_if_missing(models_list)
        total_size += get_file_size_if_missing(embedders_list)
    if exe and os.name == "nt":
        total_size += get_file_size_if_missing(executables_list)
    total_size += get_file_size_if_missing(pretraineds_hifigan)
    total_size += get_file_size_if_missing(pretraineds_refinegan_list)
    return total_size


def prequisites_download_pipeline(
    pretraineds_hifigan: bool = True,
    models: bool = True,
    exe: bool = True,
) -> None:
    """
    Manage the download pipeline for different categories of files.
    """
    total_size = calculate_total_size(
        pretraineds_hifigan_list if pretraineds_hifigan else [],
        models,
        exe,
    )

    if total_size > 0:
        with tqdm.tqdm(
            total=total_size,
            unit="iB",
            unit_scale=True,
            desc="Downloading all files",
        ) as global_bar:
            if models:
                download_mapping_files(models_list, global_bar)
                download_mapping_files(embedders_list, global_bar)
            if exe:
                if os.name == "nt":
                    download_mapping_files(executables_list, global_bar)
                else:
                    print("No executables needed")
            if pretraineds_hifigan:
                download_mapping_files(pretraineds_hifigan_list, global_bar)
                download_mapping_files(pretraineds_refinegan_list, global_bar)
