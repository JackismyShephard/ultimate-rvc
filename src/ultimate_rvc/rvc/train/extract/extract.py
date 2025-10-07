import concurrent.futures
import glob
import json
import logging
import multiprocessing as mp
import os
import sys
import time

import numpy as np
import tqdm

import torch
import torchcrepe

now_dir = os.getcwd()
sys.path.append(os.path.join(now_dir))

# Zluda hijack
import ultimate_rvc.rvc.lib.zluda
from ultimate_rvc.common import RVC_MODELS_DIR
from ultimate_rvc.rvc.configs.config import Config
from ultimate_rvc.rvc.lib.predictors.RMVPE import RMVPE0Predictor
from ultimate_rvc.rvc.lib.utils import load_audio, load_embedding
from ultimate_rvc.rvc.train.utils import remove_sox_libmso6_from_ld_preload

logger = logging.getLogger(__name__)

# Load config
config = Config()
mp.set_start_method("spawn", force=True)


class FeatureInput:

    def __init__(self, sample_rate=16000, hop_size=160, device="cpu"):
        self.fs = sample_rate
        self.hop = hop_size
        self.f0_bin = 256
        self.f0_max = 1100.0
        self.f0_min = 50.0
        self.f0_mel_min = 1127 * np.log(1 + self.f0_min / 700)
        self.f0_mel_max = 1127 * np.log(1 + self.f0_max / 700)
        self.device = device
        self.model_rmvpe = None

    def compute_f0(self, audio_array, method, hop_length):
        if method == "crepe":
            return self._get_crepe(audio_array, hop_length, type="full")
        if method == "crepe-tiny":
            return self._get_crepe(audio_array, hop_length, type="tiny")
        if method == "rmvpe":
            return self.model_rmvpe.infer_from_audio(audio_array, thred=0.03)
        raise ValueError(f"Unknown F0 method: {method}")

    def _get_crepe(self, x, hop_length, type):
        audio = torch.from_numpy(x.astype(np.float32)).to(self.device)
        audio /= torch.quantile(torch.abs(audio), 0.999)
        audio = audio.unsqueeze(0)
        pitch = torchcrepe.predict(
            audio,
            self.fs,
            hop_length,
            self.f0_min,
            self.f0_max,
            type,
            batch_size=hop_length * 2,
            device=audio.device,
            pad=True,
        )
        source = pitch.squeeze(0).cpu().float().numpy()
        source[source < 0.001] = np.nan
        return np.nan_to_num(
            np.interp(
                np.arange(0, len(source) * (x.size // self.hop), len(source))
                / (x.size // self.hop),
                np.arange(0, len(source)),
                source,
            ),
        )

    def coarse_f0(self, f0):
        f0_mel = 1127.0 * np.log(1.0 + f0 / 700.0)
        f0_mel = np.clip(
            (f0_mel - self.f0_mel_min)
            * (self.f0_bin - 2)
            / (self.f0_mel_max - self.f0_mel_min)
            + 1,
            1,
            self.f0_bin - 1,
        )
        return np.rint(f0_mel).astype(int)

    def process_file(self, file_info, f0_method, hop_length):
        inp_path, opt_path_coarse, opt_path_full, _ = file_info

        if os.path.exists(opt_path_coarse) and os.path.exists(opt_path_full):
            return

        try:
            np_arr = load_audio(inp_path, self.fs)
            feature_pit = self.compute_f0(np_arr, f0_method, hop_length)
            np.save(opt_path_full, feature_pit, allow_pickle=False)
            coarse_pit = self.coarse_f0(feature_pit)
            np.save(opt_path_coarse, coarse_pit, allow_pickle=False)
        except Exception as error:
            logger.error(  # noqa: TRY400
                "An error occurred extracting file %s on %s: %s",
                inp_path,
                self.device,
                error,
            )

    def process_files(self, files, f0_method, hop_length, device, threads):
        self.device = device
        if f0_method == "rmvpe":
            self.model_rmvpe = RMVPE0Predictor(
                os.path.join(str(RVC_MODELS_DIR), "predictors", "rmvpe.pt"),
                device=device,
            )

        def worker(file_info):
            self.process_file(file_info, f0_method, hop_length)

        with tqdm.tqdm(total=len(files), leave=True) as pbar:
            with concurrent.futures.ThreadPoolExecutor(max_workers=threads) as executor:
                futures = [executor.submit(worker, f) for f in files]
                for _ in concurrent.futures.as_completed(futures):
                    pbar.update(1)


def run_pitch_extraction(
    files: list[list[str]],
    devices: list[str],
    f0_method: str,
    hop_length: int,
    threads: int,
) -> None:
    devices_str = ", ".join(devices)
    logger.info(
        "Starting pitch extraction with %d cores on %s using %s...",
        threads,
        devices_str,
        f0_method,
    )
    start_time = time.time()
    fe = FeatureInput()

    remove_sox_libmso6_from_ld_preload()

    if f0_method == "crepe":
        actual_threads = 1  # crepe method cannot handle multiple threads
        logger.info(
            "crepe detected: using single-threaded processing (was %d threads). "
            "crepe requires single-threaded operation due to GPU limitations.",
            threads,
        )
    else:
        actual_threads = threads
    with concurrent.futures.ProcessPoolExecutor(max_workers=len(devices)) as executor:
        tasks = [
            executor.submit(
                fe.process_files,
                files[i :: len(devices)],
                f0_method,
                hop_length,
                devices[i],
                actual_threads // len(devices),
            )
            for i in range(len(devices))
        ]
        for future in concurrent.futures.as_completed(tasks):
            future.result()  # Properly waits and propagates exceptions

    logger.info("Pitch extraction completed in %.2f seconds.", time.time() - start_time)


def process_file_embedding(
    files,
    embedder_model,
    embedder_model_custom,
    device_num,
    device,
    n_threads,
):
    model = load_embedding(embedder_model, embedder_model_custom).to(device).float()
    model.eval()
    n_threads = max(1, n_threads)

    def worker(file_info):
        wav_file_path, _, _, out_file_path = file_info
        if os.path.exists(out_file_path):
            return
        feats = torch.from_numpy(load_audio(wav_file_path, 16000)).to(device).float()
        feats = feats.view(1, -1)
        with torch.no_grad():
            result = model(feats)["last_hidden_state"]
        feats_out = result.squeeze(0).float().cpu().numpy()
        if not np.isnan(feats_out).any():
            np.save(out_file_path, feats_out, allow_pickle=False)
        else:
            logger.error("%s contains NaN values and will be skipped.", wav_file_path)

    with tqdm.tqdm(total=len(files), leave=True, position=device_num) as pbar:
        with concurrent.futures.ThreadPoolExecutor(max_workers=n_threads) as executor:
            futures = [executor.submit(worker, f) for f in files]
            for _ in concurrent.futures.as_completed(futures):
                pbar.update(1)


def run_embedding_extraction(
    files: list[list[str]],
    devices: list[str],
    embedder_model: str,
    embedder_model_custom: str | None,
    threads: int,
) -> None:
    devices_str = ", ".join(devices)
    logger.info(
        "Starting embedding extraction with %d cores on %s...",
        threads,
        devices_str,
    )
    start_time = time.time()
    with concurrent.futures.ProcessPoolExecutor(max_workers=len(devices)) as executor:
        tasks = [
            executor.submit(
                process_file_embedding,
                files[i :: len(devices)],
                embedder_model,
                embedder_model_custom,
                i,
                devices[i],
                threads // len(devices),
            )
            for i in range(len(devices))
        ]
        for future in concurrent.futures.as_completed(tasks):
            future.result()  # Properly waits and propagates exceptions
    logger.info(
        "Embedding extraction completed in %.2f seconds.",
        time.time() - start_time,
    )


def initialize_extraction(
    exp_dir: str,
    f0_method: str,
    embedder_model: str,
) -> list[list[str]]:
    wav_path = os.path.join(exp_dir, "sliced_audios_16k")
    os.makedirs(os.path.join(exp_dir, f"f0_{f0_method}"), exist_ok=True)
    os.makedirs(os.path.join(exp_dir, f"f0_{f0_method}_voiced"), exist_ok=True)
    os.makedirs(
        os.path.join(exp_dir, f"{embedder_model}_extracted"),
        exist_ok=True,
    )

    files: list[list[str]] = []
    for file in glob.glob(os.path.join(wav_path, "*.wav")):
        file_name = os.path.basename(file)
        file_info = [
            file,
            os.path.join(exp_dir, f"f0_{f0_method}", file_name + ".npy"),
            os.path.join(exp_dir, f"f0_{f0_method}_voiced", file_name + ".npy"),
            os.path.join(
                exp_dir,
                f"{embedder_model}_extracted",
                file_name.replace("wav", "npy"),
            ),
        ]
        files.append(file_info)

    return files


def update_model_info(
    exp_dir: str,
    embedder_model: str,
    custom_embedder_model_hash: str | None,
) -> None:
    file_path = os.path.join(exp_dir, "model_info.json")
    if os.path.exists(file_path):
        with open(file_path) as f:
            data = json.load(f)
    else:
        data = {}
    data["embedder_model"] = embedder_model
    data["custom_embedder_model_hash"] = custom_embedder_model_hash
    with open(file_path, "w") as f:
        json.dump(data, f, indent=4)
