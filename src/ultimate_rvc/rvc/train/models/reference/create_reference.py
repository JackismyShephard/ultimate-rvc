import logging

import numpy as np

import torch
from transformers import HubertModel

import librosa

from ultimate_rvc.common import EMBEDDER_MODELS_DIR
from ultimate_rvc.rvc.common import RVC_TRAINING_MODELS_DIR
from ultimate_rvc.rvc.lib.predictors.f0 import RMVPE

logger = logging.getLogger(__name__)


def cf0(f0):
    """Convert F0 to coarse F0."""
    f0_bin = 256
    f0_max = 1100.0
    f0_min = 50.0
    f0_mel_min = 1127 * np.log(1 + f0_min / 700)
    f0_mel_max = 1127 * np.log(1 + f0_max / 700)
    f0_mel = 1127 * np.log(1 + f0 / 700)
    f0_mel = np.clip(
        (f0_mel - f0_mel_min) * (f0_bin - 2) / (f0_mel_max - f0_mel_min) + 1,
        1,
        f0_bin - 1,
    )
    return np.rint(f0_mel).astype(int)


ref = r"reference.wav"
audio, sr = librosa.load(ref, sr=16000)
trimmed_len = (len(audio) // 320) * 320
# to prevent feature and pitch offset mismatch
audio = audio[:trimmed_len]

logger.info("audio", audio.shape)
rmvpe_model = RMVPE(device="cpu", sample_rate=16000, hop_size=160)
f0 = rmvpe_model.get_f0(audio, filter_radius=0.03)
logger.info("f0", f0.shape)
f0c = cf0(f0)
logger.info("f0c", f0c.shape)

cv_path = EMBEDDER_MODELS_DIR / "contentvec"
cv_model = HubertModel.from_pretrained(cv_path)

spin_path = EMBEDDER_MODELS_DIR / "spin"
spin_model = HubertModel.from_pretrained(spin_path)

spin2_path = EMBEDDER_MODELS_DIR / "spin-v2"
spin2_model = HubertModel.from_pretrained(spin2_path)

feats = torch.from_numpy(audio).to(torch.float32).to("cpu")
feats = torch.nn.functional.pad(feats.unsqueeze(0), (40, 40), mode="reflect")
feats = feats.view(1, -1)

with torch.no_grad():
    cv_feats = cv_model(feats)["last_hidden_state"]
    cv_feats = cv_feats.squeeze(0).float().cpu().numpy()
    print("cv", cv_feats.shape)

    spin_feats = spin_model(feats)["last_hidden_state"]
    spin_feats = spin_feats.squeeze(0).float().cpu().numpy()
    print("spin", spin_feats.shape)

    spin2_feats = spin2_model(feats)["last_hidden_state"]
    spin2_feats = spin2_feats.squeeze(0).float().cpu().numpy()
    print("spin-v2", spin2_feats.shape)

np.save(RVC_TRAINING_MODELS_DIR / "reference" / "contentvec" / "feats.npy", cv_feats)
np.save(RVC_TRAINING_MODELS_DIR / "reference" / "spin" / "feats.npy", spin_feats)
np.save(RVC_TRAINING_MODELS_DIR / "reference" / "spin-v2" / "feats.npy", spin2_feats)
np.save(RVC_TRAINING_MODELS_DIR / "reference" / "pitch_coarse.npy", f0c)
np.save(RVC_TRAINING_MODELS_DIR / "reference" / "pitch_fine.npy", f0)
