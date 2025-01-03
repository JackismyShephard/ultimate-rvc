import json
import os
import shutil
from random import shuffle

from ultimate_rvc.rvc.common import RVC_CONFIGS_DIR, RVC_TRAINING_MODELS_DIR
from ultimate_rvc.rvc.configs.config import Config

config = Config()
current_directory = os.getcwd()


def generate_config(rvc_version: str, model_path: str):
    file_path = os.path.join(model_path, "model_info.json")
    with open(file_path) as f:
        data = json.load(f)
    sample_rate = data["sample_rate"]
    config_path = os.path.join(RVC_CONFIGS_DIR, rvc_version, f"{sample_rate}.json")
    config_save_path = os.path.join(model_path, "config.json")
    shutil.copyfile(config_path, config_save_path)


def generate_filelist(
    model_path: str,
    rvc_version: str,
    include_mutes: int,
    f0_method_id: str,
    embedder_model_id: str,
):
    file_path = os.path.join(model_path, "model_info.json")
    with open(file_path) as f:
        data = json.load(f)
    sample_rate = data["sample_rate"]

    gt_wavs_dir = os.path.join(model_path, "sliced_audios")
    feature_dir = os.path.join(
        model_path,
        f"{rvc_version}_{embedder_model_id}_extracted",
    )

    f0_dir, f0nsf_dir = None, None
    f0_dir = os.path.join(model_path, f"f0_{f0_method_id}")
    f0nsf_dir = os.path.join(model_path, f"f0_{f0_method_id}_voiced")

    gt_wavs_files = set(name.split(".")[0] for name in os.listdir(gt_wavs_dir))
    feature_files = set(name.split(".")[0] for name in os.listdir(feature_dir))

    f0_files = set(name.split(".")[0] for name in os.listdir(f0_dir))
    f0nsf_files = set(name.split(".")[0] for name in os.listdir(f0nsf_dir))
    names = gt_wavs_files & feature_files & f0_files & f0nsf_files

    options = []
    mute_base_path = os.path.join(RVC_TRAINING_MODELS_DIR, "mute")
    sids = []
    for name in names:
        sid = name.split("_")[0]
        if sid not in sids:
            sids.append(sid)
        options.append(
            f"{gt_wavs_dir}/{name}.wav|{feature_dir}/{name}.npy|{f0_dir}/{name}.wav.npy|{f0nsf_dir}/{name}.wav.npy|{sid}",
        )
    if include_mutes > 0:
        mute_audio_path = os.path.join(
            mute_base_path,
            "sliced_audios",
            f"mute{sample_rate}.wav",
        )
        mute_feature_path = os.path.join(
            mute_base_path,
            f"{rvc_version}_extracted",
            "mute.npy",
        )
        mute_f0_path = os.path.join(mute_base_path, "f0", "mute.wav.npy")
        mute_f0nsf_path = os.path.join(mute_base_path, "f0_voiced", "mute.wav.npy")

        # adding x files per sid
        for sid in sids * include_mutes:
            options.append(
                f"{mute_audio_path}|{mute_feature_path}|{mute_f0_path}|{mute_f0nsf_path}|{sid}",
            )

    data.update(
        {
            "speakers_id": len(sids),
        },
    )
    with open(file_path, "w") as f:
        json.dump(data, f, indent=4)

    shuffle(options)

    with open(os.path.join(model_path, "filelist.txt"), "w") as f:
        f.write("\n".join(options))
