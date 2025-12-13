import datetime
import glob
import json
import logging
import os
import pathlib
import shutil
import signal
import sys
from collections import deque
from random import randint, shuffle
from time import time as ttime

import numpy as np
from tqdm import tqdm

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

now_dir = pathlib.Path.cwd()
sys.path.append(os.path.join(str(now_dir)))

# Zluda hijack
import ultimate_rvc.rvc.lib.zluda
from ultimate_rvc.common import TRAINING_MODELS_DIR
from ultimate_rvc.rvc.common import RVC_TRAINING_MODELS_DIR
from ultimate_rvc.rvc.lib.algorithm import commons
from ultimate_rvc.rvc.train.losses import (
    discriminator_loss,
    feature_loss,
    generator_loss,
    kl_loss,
)
from ultimate_rvc.rvc.train.mel_processing import (
    MultiScaleMelSpectrogramLoss,
    mel_spectrogram_torch,
    spec_to_mel_torch,
)
from ultimate_rvc.rvc.train.process.extract_model import extract_model
from ultimate_rvc.rvc.train.utils import (
    HParams,
    latest_checkpoint_path,
    load_checkpoint,
    load_wav_to_torch,
    plot_spectrogram_to_numpy,
    remove_sox_libmso6_from_ld_preload,
    save_checkpoint,
    summarize,
)

logging.getLogger("torch").setLevel(logging.ERROR)
logger = logging.getLogger(__name__)

torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True
torch.multiprocessing.set_start_method("spawn", force=True)
os.environ["USE_LIBUV"] = "0" if sys.platform == "win32" else "1"

randomized = True
d_lr_coeff = 1.0
g_lr_coeff = 1.0
d_step_per_g_step = 1
bf16_adamw = False
global_step = 0
lowest_g_value = {"value": float("inf"), "epoch": 0}
lowest_d_value = {"value": float("inf"), "epoch": 0}
consecutive_increases_gen = 0
consecutive_increases_disc = 0

avg_losses = {
    "grad_d_50": deque(maxlen=50),
    "grad_g_50": deque(maxlen=50),
    "disc_loss_50": deque(maxlen=50),
    "adv_loss_50": deque(maxlen=50),
    "fm_loss_50": deque(maxlen=50),
    "kl_loss_50": deque(maxlen=50),
    "mel_loss_50": deque(maxlen=50),
    "gen_loss_50": deque(maxlen=50),
}


class EpochRecorder:
    """
    Records the time elapsed per epoch.
    """

    def __init__(self):
        self.last_time = ttime()

    def record(self):
        """
        Records the elapsed time and returns a formatted string.
        """
        now_time = ttime()
        elapsed_time = now_time - self.last_time
        self.last_time = now_time
        elapsed_time = round(elapsed_time, 1)
        elapsed_time_str = str(datetime.timedelta(seconds=int(elapsed_time)))
        current_time = datetime.datetime.now().strftime("%H:%M:%S")
        return f"time={current_time} | training_speed={elapsed_time_str}"


def main(
    model_name: str,
    sample_rate: int | None,
    vocoder: str,
    total_epoch: int,
    batch_size: int,
    save_every_epoch: int,
    save_only_latest: bool,
    save_every_weights: bool,
    pretrain_g: str,
    pretrain_d: str,
    overtraining_detector: bool,
    overtraining_threshold: int,
    cleanup: bool,
    cache_data_in_gpu: bool,
    checkpointing: bool,
    device_type: str,
    gpus: set[int] | None,
    precision: str = "fp32",
) -> None:
    """
    Start the training process.

    Raises:
        RuntimeError: If the sample rate of the pretrained model does not match the dataset audio sample rate.

    """
    remove_sox_libmso6_from_ld_preload()
    experiment_dir = os.path.join(TRAINING_MODELS_DIR, model_name)
    config_save_path = os.path.join(experiment_dir, "config.json")

    # Use a Manager to create a shared list
    manager = mp.Manager()
    global_gen_loss = manager.list([0] * total_epoch)
    global_disc_loss = manager.list([0] * total_epoch)

    try:
        with pathlib.Path(config_save_path).open() as f:
            config = json.load(f)
        config = HParams(**config)
    except FileNotFoundError:
        logger.error(
            "Config file not found at %s. Did you run preprocessing and feature"
            " extraction steps?",
            config_save_path,
        )
        sys.exit(1)
    sample_rate = config.data.sample_rate if sample_rate is None else sample_rate

    if (
        precision == "bf16"
        and torch.cuda.is_available()
        and torch.cuda.is_bf16_supported()
    ):
        train_dtype = torch.bfloat16
    elif precision == "fp16" and torch.cuda.is_available():
        train_dtype = torch.float16
    else:
        train_dtype = torch.float32

    config.data.training_files = os.path.join(experiment_dir, "filelist.txt")

    # Set up distributed training environment for master node.
    # master node is localhost because we are running on a single local
    # machine. master port is randomly selected
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(randint(20000, 55555))
    logger.info("MASTER_PORT: %s", os.environ["MASTER_PORT"])
    # Check sample rate
    wavs = glob.glob(
        os.path.join(experiment_dir, "sliced_audios", "*.wav"),
    )
    if wavs:
        _, sr = load_wav_to_torch(wavs[0])
        if sr != sample_rate:
            error_message = (
                f"Error: Pretrained model sample rate ({sample_rate} Hz) does not match"
                f" dataset audio sample rate ({sr} Hz)."
            )
            raise RuntimeError(error_message)
    else:
        logger.warning("No wav file found.")

    device = torch.device(device_type)
    gpus = gpus or {0}
    n_gpus = len(gpus)

    if device.type == "cpu":
        logger.warning("Training with CPU, this will take a long time.")

    def start() -> None:
        """Start the training process with multi-GPU support or CPU."""
        children = []
        pid_data = {"process_pids": []}
        with pathlib.Path(config_save_path).open("r") as pid_file:
            try:
                existing_data = json.load(pid_file)
                pid_data.update(existing_data)
            except json.JSONDecodeError:
                pass
        with pathlib.Path(config_save_path).open("w") as pid_file:
            for rank, device_id in enumerate(gpus):
                subproc = mp.Process(
                    target=run,
                    args=(
                        rank,
                        n_gpus,
                        experiment_dir,
                        pretrain_g,
                        pretrain_d,
                        total_epoch,
                        save_every_weights,
                        config,
                        device,
                        device_id,
                        model_name,
                        sample_rate,
                        vocoder,
                        batch_size,
                        save_every_epoch,
                        save_only_latest,
                        overtraining_detector,
                        overtraining_threshold,
                        checkpointing,
                        cache_data_in_gpu,
                        global_gen_loss,
                        global_disc_loss,
                        train_dtype,
                    ),
                )
                children.append(subproc)
                subproc.start()
                pid_data["process_pids"].append(subproc.pid)
            json.dump(pid_data, pid_file, indent=4)
        cancel_signal = signal.SIGTERM if os.name == "nt" else -signal.SIGTERM
        error_codes = []
        for i in range(n_gpus):
            children[i].join()
            exit_code = children[i].exitcode
            if exit_code != 0:
                logger.warning(
                    "Process running on device %s exited with code %s.",
                    device_id,
                    exit_code,
                )
                if exit_code != cancel_signal:
                    error_codes.append(exit_code)
        if error_codes:
            err_msg = (
                "One or more training processes failed. See the logs or console for"
                " details."
            )
            raise RuntimeError(err_msg)

    if cleanup:
        logger.info("Removing files from the prior training attempt...")

        # Clean up unnecessary files
        for entry in os.scandir(os.path.join(TRAINING_MODELS_DIR, model_name)):
            if entry.is_file():
                _, file_extension = os.path.splitext(entry.name)
                if file_extension in {".0", ".pth", ".index"}:
                    pathlib.Path(entry.path).unlink()
            elif entry.is_dir() and entry.name == "eval":
                shutil.rmtree(entry.path)

        logger.info("Cleanup done!")
    start()


def run(
    rank,
    n_gpus,
    experiment_dir,
    pretrain_g,
    pretrain_d,
    custom_total_epoch,
    custom_save_every_weights,
    config,
    device,
    device_id,
    model_name,
    sample_rate,
    vocoder,
    batch_size,
    save_every_epoch,
    save_only_latest,
    overtraining_detector,
    overtraining_threshold,
    checkpointing,
    cache_data_in_gpu,
    global_gen_loss,
    global_disc_loss,
    train_dtype,
):
    """
    Runs the training loop on a specific GPU or CPU.

    Args:
        rank (int): The rank of the current process within the distributed training setup.
        n_gpus (int): The total number of GPUs available for training.
        experiment_dir (str): The directory where experiment logs and checkpoints will be saved.
        pretrain_g (str): Path to the pre-trained generator model.
        pretrain_d (str): Path to the pre-trained discriminator model.
        custom_total_epoch (int): The total number of epochs for training.
        custom_save_every_weights (int): The interval (in epochs) at which to save model weights.
        config (object): Configuration object containing training parameters.
        device (torch.device): The device to use for training (CPU or GPU).

    """
    global global_step, optimizer, lowest_d_value, lowest_g_value, consecutive_increases_gen, consecutive_increases_disc

    if rank == 0:
        writer_eval = SummaryWriter(log_dir=os.path.join(experiment_dir, "eval"))
    else:
        writer_eval = None

    # Initialize distributed training environment for child node.
    dist.init_process_group(
        backend="gloo" if sys.platform == "win32" or device.type != "cuda" else "nccl",
        init_method="env://",
        world_size=n_gpus if device.type == "cuda" else 1,
        rank=rank if device.type == "cuda" else 0,
    )

    torch.manual_seed(config.train.seed)

    if device.type == "cuda":
        torch.cuda.set_device(device_id)

    # Create datasets and dataloaders
    from ultimate_rvc.rvc.train.data_utils import (
        DistributedBucketSampler,
        TextAudioCollateMultiNSFsid,
        TextAudioLoaderMultiNSFsid,
    )

    train_dataset = TextAudioLoaderMultiNSFsid(config.data)
    collate_fn = TextAudioCollateMultiNSFsid()
    train_sampler = DistributedBucketSampler(
        train_dataset,
        batch_size * n_gpus,
        [50, 100, 200, 300, 400, 500, 600, 700, 800, 900],
        num_replicas=n_gpus,
        rank=rank,
        shuffle=True,
    )

    train_loader = DataLoader(
        train_dataset,
        num_workers=4,
        shuffle=False,
        pin_memory=True,
        collate_fn=collate_fn,
        batch_sampler=train_sampler,
        persistent_workers=True,
        prefetch_factor=8,
    )
    if len(train_loader) < 3:
        logger.error(
            "Not enough data in the training set. Perhaps you forgot to slice the"
            " audio files in preprocess?",
        )
        sys.exit(1)

    # defaults
    embedder_name = "contentvec"
    spk_dim = config.model.spk_embed_dim  # 109 default speakers

    model_info_path = os.path.join(experiment_dir, "model_info.json")
    try:
        with pathlib.Path(model_info_path).open("r") as f:
            model_info = json.load(f)
            embedder_name = model_info["embedder_model"]
            spk_dim = model_info["speakers_id"]
    except Exception as e:
        logger.error("Could not load model info file: %s. Using defaults.", e)

    # Try to load speaker dim from latest checkpoint or pretrain_g
    try:
        last_g = latest_checkpoint_path(experiment_dir, "G_*.pth")
        chk_path = last_g or (pretrain_g if pretrain_g not in {"", "None"} else None)

        if chk_path:
            ckpt = torch.load(chk_path, map_location="cpu", weights_only=False)
            spk_dim = ckpt["model"]["emb_g.weight"].shape[0]
            del ckpt
    except Exception as e:
        logger.error(
            "Failed to load checkpoint: %s. Using default number of speakers.", e
        )

    # update config before the model init
    logger.info("Initializing the generator with %s speakers.", spk_dim)
    config.model.spk_embed_dim = spk_dim

    # Initialize models and optimizers
    from ultimate_rvc.rvc.lib.algorithm.discriminators import MultiPeriodDiscriminator
    from ultimate_rvc.rvc.lib.algorithm.synthesizers import Synthesizer

    # NOTE checkingpointing here means whether or not activations are
    # saved during forward pass for backpropagation during backward pass

    net_g = Synthesizer(
        config.data.filter_length // 2 + 1,
        config.train.segment_size // config.data.hop_length,
        **config.model,
        use_f0=True,
        sr=sample_rate,
        vocoder=vocoder,
        checkpointing=checkpointing,
        randomized=randomized,
    )
    if vocoder == "RefineGAN":
        disc_version = "v3"
        fn_mel_loss = MultiScaleMelSpectrogramLoss(sample_rate=sample_rate)
        logger.info("Using Multi-Scale Mel loss function")
    else:
        disc_version = "v2"
        fn_mel_loss = torch.nn.L1Loss()
        logger.info("Using Single-Scale Mel loss function")
    net_d = MultiPeriodDiscriminator(
        config.model.use_spectral_norm,
        checkpointing=checkpointing,
        version=disc_version,
    )

    if device.type == "cuda":
        net_g = net_g.cuda(device_id)
        net_d = net_d.cuda(device_id)
    else:
        net_g = net_g.to(device)
        net_d = net_d.to(device)

    if bf16_adamw and train_dtype == torch.bfloat16:
        logger.info("Using BFloat16 AdamW optimizer")
        from ultimate_rvc.rvc.train.anyprecision_optimizer import AnyPrecisionAdamW

        optimizer = AnyPrecisionAdamW
    else:
        logger.info("Using AdamW optimizer")
        optimizer = torch.optim.AdamW

    optim_g = optimizer(
        net_g.parameters(),
        config.train.learning_rate * g_lr_coeff,
        betas=config.train.betas,
        eps=config.train.eps,
    )
    optim_d = optimizer(
        net_d.parameters(),
        config.train.learning_rate * d_lr_coeff,
        betas=config.train.betas,
        eps=config.train.eps,
    )

    # Wrap models with DDP for multi-gpu processing
    if n_gpus > 1 and device.type == "cuda":
        net_g = DDP(net_g, device_ids=[device_id])
        net_d = DDP(net_d, device_ids=[device_id])

    if rank == 0 and device.type == "cuda" and train_dtype == torch.bfloat16:
        logger.info("Using BFloat16 for training.")
    elif rank == 0 and device.type == "cuda" and train_dtype == torch.float16:
        logger.info("Using Float16 for training.")

    # Load checkpoint if available
    scaler_dict = {}
    try:
        _, _, _, epoch_str, lowest_d_value, consecutive_increases_disc, scaler_dict = (
            load_checkpoint(
                latest_checkpoint_path(experiment_dir, "D_*.pth"),
                net_d,
                optim_d,
            )
        )
        _, _, _, epoch_str, lowest_g_value, consecutive_increases_gen, _ = (
            load_checkpoint(
                latest_checkpoint_path(experiment_dir, "G_*.pth"),
                net_g,
                optim_g,
            )
        )
        epoch_str += 1
        global_step = (epoch_str - 1) * len(train_loader)
        logger.info("Resumed from epoch %s", epoch_str)
        logger.info(
            "Loaded lowest generator loss %.3f at epoch %s, lowest discriminator loss"
            " %.3f at epoch %s",
            lowest_g_value["value"],
            lowest_g_value["epoch"],
            lowest_d_value["value"],
            lowest_d_value["epoch"],
        )
        logger.info(
            "Loaded consecutive increases gen %d, consecutive increases disc %d",
            consecutive_increases_gen,
            consecutive_increases_disc,
        )

    except Exception:
        epoch_str = 1
        global_step = 0
        if pretrain_g not in {"", "None"}:
            if rank == 0:
                logger.info("Loaded pretrained (G) '%s'", pretrain_g)
            try:
                ckpt = torch.load(pretrain_g, map_location="cpu", weights_only=False)[
                    "model"
                ]
                if hasattr(net_g, "module"):
                    net_g.module.load_state_dict(ckpt)
                else:
                    net_g.load_state_dict(ckpt)
                del ckpt
            except RuntimeError:
                logger.error(  # noqa: TRY400
                    "The parameters of the pretrain model such as the sample rate or"
                    " architecture do not match the selected model.",
                )
                sys.exit(1)

        if pretrain_d not in {"", "None"}:
            if rank == 0:
                logger.info("Loaded pretrained (D) '%s'", pretrain_d)
            try:
                ckpt = torch.load(pretrain_d, map_location="cpu", weights_only=False)[
                    "model"
                ]
                if hasattr(net_d, "module"):
                    net_d.module.load_state_dict(ckpt)
                else:
                    net_d.load_state_dict(ckpt)
                del ckpt
            except RuntimeError:
                logger.error(  # noqa: TRY400
                    "The parameters of the pretrain model such as the sample rate or"
                    " architecture do not match the selected model.",
                )
                sys.exit(1)

    # Initialize schedulers
    scheduler_g = torch.optim.lr_scheduler.ExponentialLR(
        optim_g,
        gamma=config.train.lr_decay,
        last_epoch=epoch_str - 2,
    )
    scheduler_d = torch.optim.lr_scheduler.ExponentialLR(
        optim_d,
        gamma=config.train.lr_decay,
        last_epoch=epoch_str - 2,
    )

    use_scaler = device.type == "cuda" and train_dtype == torch.float16
    scaler = torch.amp.GradScaler(enabled=use_scaler)
    if len(scaler_dict) > 0:
        scaler.load_state_dict(scaler_dict)

    cache = []
    # collect the reference audio for tensorboard evaluation
    if pathlib.Path(
        os.path.join(RVC_TRAINING_MODELS_DIR, "reference", embedder_name, "feats.npy")
    ).is_file():
        logger.info("Using %s reference set for validation", embedder_name)
        phone = np.load(
            os.path.join(
                RVC_TRAINING_MODELS_DIR,
                "reference",
                embedder_name,
                "feats.npy",
            ),
        )
        # expanding x2 to match pitch size
        phone = np.repeat(phone, 2, axis=0)
        phone_lengths = torch.LongTensor([phone.shape[0]]).to(device)
        phone = torch.FloatTensor(phone).unsqueeze(0).to(device)
        pitch = np.load(
            os.path.join(
                RVC_TRAINING_MODELS_DIR,
                "reference",
                "pitch_coarse.npy",
            ),
        )
        # removed last frame to match features
        pitch = torch.LongTensor(pitch[:-1]).unsqueeze(0).to(device)
        pitchf = np.load(
            os.path.join(
                RVC_TRAINING_MODELS_DIR,
                "reference",
                "pitch_fine.npy",
            ),
        )
        # removed last frame to match features
        pitchf = torch.FloatTensor(pitchf[:-1]).unsqueeze(0).to(device)
        sid = torch.LongTensor([0]).to(device)
        reference = (
            phone,
            phone_lengths,
            pitch,
            pitchf,
            sid,
        )
    else:
        logger.info(
            "No custom reference found, using a default audio sample for validation"
        )
        info = next(iter(train_loader))
        phone, phone_lengths, pitch, pitchf, _, _, _, _, sid = info
        reference = (
            phone.to(device),
            phone_lengths.to(device),
            pitch.to(device),
            pitchf.to(device),
            sid.to(device),
        )
    if epoch_str > custom_total_epoch:
        cleanup_training_processes(experiment_dir)
        return
    logger.info("Starting training...")
    for epoch in range(epoch_str, custom_total_epoch + 1):
        train_and_evaluate(
            rank,
            epoch,
            config,
            [net_g, net_d],
            [optim_g, optim_d],
            [scheduler_g, scheduler_d],
            [train_loader, None],
            [writer_eval],
            cache,
            custom_save_every_weights,
            custom_total_epoch,
            device,
            device_id,
            reference,
            fn_mel_loss,
            model_name,
            experiment_dir,
            sample_rate,
            vocoder,
            save_every_epoch,
            save_only_latest,
            overtraining_detector,
            overtraining_threshold,
            cache_data_in_gpu,
            global_gen_loss,
            global_disc_loss,
            scaler,
            train_dtype,
        )


def train_and_evaluate(
    rank,
    epoch,
    config,
    nets,
    optims,
    schedulers,
    loaders,
    writers,
    cache,
    custom_save_every_weights,
    custom_total_epoch,
    device,
    device_id,
    reference,
    fn_mel_loss,
    model_name,
    experiment_dir,
    sample_rate,
    vocoder,
    save_every_epoch,
    save_only_latest,
    overtraining_detector,
    overtraining_threshold,
    cache_data_in_gpu,
    global_gen_loss,
    global_disc_loss,
    scaler,
    train_dtype,
) -> None:
    """Train and evaluates the model for one epoch."""
    global global_step, lowest_g_value, lowest_d_value, consecutive_increases_gen, consecutive_increases_disc

    model_add = []
    checkpoint_idxs = []
    done = False

    net_g, net_d = nets
    optim_g, optim_d = optims
    scheduler_g, scheduler_d = schedulers
    train_loader = loaders[0] if loaders is not None else None
    if writers is not None:
        writer = writers[0]

    train_loader.batch_sampler.set_epoch(epoch)

    net_g.train()
    net_d.train()

    use_amp = device.type == "cuda" and (train_dtype in {torch.bfloat16, torch.float16})

    # Data caching
    if device.type == "cuda" and cache_data_in_gpu:
        if cache == []:
            for batch_idx, info in enumerate(train_loader):
                # phone, phone_lengths, pitch, pitchf, spec, spec_lengths, wave, wave_lengths, sid
                info = [tensor.cuda(device_id, non_blocking=True) for tensor in info]
                cache.append((batch_idx, info))
        shuffle(cache)
        data_iterator = cache
    else:
        data_iterator = enumerate(train_loader)

    epoch_recorder = EpochRecorder()
    with tqdm(total=len(train_loader), leave=False) as pbar:
        for batch_idx, info in data_iterator:
            if device.type == "cuda" and not cache_data_in_gpu:
                info = [tensor.cuda(device_id, non_blocking=True) for tensor in info]
            elif device.type != "cuda":
                info = [tensor.to(device) for tensor in info]
            # else iterator is going thru a cached list with a device already assigned

            (
                phone,
                phone_lengths,
                pitch,
                pitchf,
                spec,
                spec_lengths,
                wave,
                wave_lengths,
                sid,
            ) = info

            with torch.amp.autocast(
                device_type="cuda", enabled=use_amp, dtype=train_dtype
            ):
                # Forward pass
                model_output = net_g(
                    phone,
                    phone_lengths,
                    pitch,
                    pitchf,
                    spec,
                    spec_lengths,
                    sid,
                )
                y_hat, ids_slice, x_mask, z_mask, (z, z_p, m_p, logs_p, m_q, logs_q) = (
                    model_output
                )
                # slice of the original waveform to match a generate slice
                if randomized:
                    wave = commons.slice_segments(
                        wave,
                        ids_slice * config.data.hop_length,
                        config.train.segment_size,
                        dim=3,
                    )
            for _ in range(d_step_per_g_step):  # default x1
                with torch.amp.autocast(
                    device_type="cuda", enabled=use_amp, dtype=train_dtype
                ):
                    y_d_hat_r, y_d_hat_g, _, _ = net_d(wave, y_hat.detach())
                loss_disc, _, _ = discriminator_loss(y_d_hat_r, y_d_hat_g)
                # Discriminator backward and update
                global_disc_loss[epoch - 1] += loss_disc.item()
                optim_d.zero_grad()
                if device.type == "cuda" and train_dtype == torch.float16:
                    scaler.scale(loss_disc).backward()
                    scaler.unscale_(optim_d)
                    grad_norm_d = commons.grad_norm(net_d.parameters())
                    scaler.step(optim_d)
                else:
                    loss_disc.backward()
                    grad_norm_d = commons.grad_norm(net_d.parameters())
                    optim_d.step()

            with torch.amp.autocast(
                device_type="cuda", enabled=use_amp, dtype=train_dtype
            ):
                # Generator backward and update
                _, y_d_hat_g, fmap_r, fmap_g = net_d(wave, y_hat)

            if vocoder == "RefineGAN":
                loss_mel = fn_mel_loss(wave, y_hat) * config.train.c_mel / 3.0
            else:
                wave_mel = mel_spectrogram_torch(
                    wave.float().squeeze(1),
                    config.data.filter_length,
                    config.data.n_mel_channels,
                    config.data.sample_rate,
                    config.data.hop_length,
                    config.data.win_length,
                    config.data.mel_fmin,
                    config.data.mel_fmax,
                )
                y_hat_mel = mel_spectrogram_torch(
                    y_hat.float().squeeze(1),
                    config.data.filter_length,
                    config.data.n_mel_channels,
                    config.data.sample_rate,
                    config.data.hop_length,
                    config.data.win_length,
                    config.data.mel_fmin,
                    config.data.mel_fmax,
                )
                loss_mel = fn_mel_loss(wave_mel, y_hat_mel) * config.train.c_mel
            loss_kl = kl_loss(z_p, logs_q, m_p, logs_p, z_mask) * config.train.c_kl
            loss_fm = feature_loss(fmap_r, fmap_g)
            loss_gen, _ = generator_loss(y_d_hat_g)
            loss_gen_all = loss_gen + loss_fm + loss_mel + loss_kl
            global_gen_loss[epoch - 1] += loss_gen_all.item()
            optim_g.zero_grad()
            if device.type == "cuda" and train_dtype == torch.float16:
                scaler.scale(loss_gen_all).backward()
                scaler.unscale_(optim_g)
                grad_norm_g = commons.grad_norm(net_g.parameters())
                scaler.step(optim_g)
                scaler.update()
            else:
                loss_gen_all.backward()
                grad_norm_g = commons.grad_norm(net_g.parameters())
                optim_g.step()

            global_step += 1

            # queue for rolling losses over 50 steps
            avg_losses["grad_d_50"].append(grad_norm_d)
            avg_losses["grad_g_50"].append(grad_norm_g)
            avg_losses["disc_loss_50"].append(loss_disc.detach())
            avg_losses["adv_loss_50"].append(loss_gen.detach())
            avg_losses["fm_loss_50"].append(loss_fm.detach())
            avg_losses["kl_loss_50"].append(loss_kl.detach())
            avg_losses["mel_loss_50"].append(loss_mel.detach())
            avg_losses["gen_loss_50"].append(loss_gen_all.detach())

            if rank == 0 and global_step % 50 == 0:
                # logging rolling averages
                scalar_dict = {
                    "grad_avg_50/norm_d": (
                        sum(avg_losses["grad_d_50"]) / len(avg_losses["grad_d_50"])
                    ),
                    "grad_avg_50/norm_g": (
                        sum(avg_losses["grad_g_50"]) / len(avg_losses["grad_g_50"])
                    ),
                    "loss_avg_50/d/adv": torch.mean(
                        torch.stack(list(avg_losses["disc_loss_50"])),
                    ),
                    "loss_avg_50/g/adv": torch.mean(
                        torch.stack(list(avg_losses["adv_loss_50"])),
                    ),
                    "loss_avg_50/g/fm": torch.mean(
                        torch.stack(list(avg_losses["fm_loss_50"])),
                    ),
                    "loss_avg_50/g/kl": torch.mean(
                        torch.stack(list(avg_losses["kl_loss_50"])),
                    ),
                    "loss_avg_50/g/mel": torch.mean(
                        torch.stack(list(avg_losses["mel_loss_50"])),
                    ),
                    "loss_avg_50/g/total": torch.mean(
                        torch.stack(list(avg_losses["gen_loss_50"])),
                    ),
                }
                summarize(
                    writer=writer,
                    global_step=global_step,
                    scalars=scalar_dict,
                )

            pbar.update(1)
        # end of batch train
    # end of tqdm
    scheduler_d.step()
    scheduler_g.step()

    with torch.no_grad():
        torch.cuda.empty_cache()
    # Logging and checkpointing
    if rank == 0:
        avg_global_disc_loss = global_disc_loss[epoch - 1] / len(train_loader.dataset)
        avg_global_gen_loss = global_gen_loss[epoch - 1] / len(train_loader.dataset)

        min_delta = 0.004

        if avg_global_disc_loss < lowest_d_value["value"] - min_delta:
            lowest_d_value = {"value": avg_global_disc_loss, "epoch": epoch}
            consecutive_increases_disc = 0
        else:
            consecutive_increases_disc += 1

        if avg_global_gen_loss < lowest_g_value["value"] - min_delta:
            logger.info(
                "New best epoch %d with average generator loss %.3f and discriminator"
                " loss %.3f",
                epoch,
                avg_global_gen_loss,
                avg_global_disc_loss,
            )
            lowest_g_value = {"value": avg_global_gen_loss, "epoch": epoch}
            consecutive_increases_gen = 0
            model_add.append(
                os.path.join(experiment_dir, f"{model_name}_best.pth"),
            )
        else:
            consecutive_increases_gen += 1

        # used for tensorboard chart - all/mel
        mel = spec_to_mel_torch(
            spec,
            config.data.filter_length,
            config.data.n_mel_channels,
            config.data.sample_rate,
            config.data.mel_fmin,
            config.data.mel_fmax,
        )
        # used for tensorboard chart - slice/mel_org
        if randomized:
            y_mel = commons.slice_segments(
                mel,
                ids_slice,
                config.train.segment_size // config.data.hop_length,
                dim=3,
            )
        else:
            y_mel = mel
        # used for tensorboard chart - slice/mel_gen
        y_hat_mel = mel_spectrogram_torch(
            y_hat.float().squeeze(1),
            config.data.filter_length,
            config.data.n_mel_channels,
            config.data.sample_rate,
            config.data.hop_length,
            config.data.win_length,
            config.data.mel_fmin,
            config.data.mel_fmax,
        )

        lr = optim_g.param_groups[0]["lr"]

        scalar_dict = {
            "loss/g/total": loss_gen_all,
            "loss/d/adv": loss_disc,
            "learning_rate": lr,
            "grad/norm_d": grad_norm_d,
            "grad/norm_g": grad_norm_g,
            "loss/g/adv": loss_gen,
            "loss/g/fm": loss_fm,
            "loss/g/mel": loss_mel,
            "loss/g/kl": loss_kl,
        }

        image_dict = {
            "slice/mel_org": plot_spectrogram_to_numpy(y_mel[0].data.cpu().numpy()),
            "slice/mel_gen": plot_spectrogram_to_numpy(y_hat_mel[0].data.cpu().numpy()),
            "all/mel": plot_spectrogram_to_numpy(mel[0].data.cpu().numpy()),
        }
        overtrain_info = ""
        # Print training progress
        lowest_g_value_rounded = float(lowest_g_value["value"])
        lowest_g_value_rounded = round(lowest_g_value_rounded, 3)

        record = f"{model_name} | epoch={epoch} | {epoch_recorder.record()}"
        record += (
            f" | best avg-gen-loss={lowest_g_value_rounded:.3f} (epoch"
            f" {lowest_g_value['epoch']})"
        )
        # Check overtraining
        if overtraining_detector:
            overtrain_info = (
                f"Average epoch generator loss {avg_global_gen_loss:.3f} and"
                f" discriminator loss {avg_global_disc_loss:.3f}"
            )

            remaining_epochs_gen = max(
                overtraining_threshold - consecutive_increases_gen,
                0,
            )
            remaining_epochs_disc = max(
                overtraining_threshold * 2 - consecutive_increases_disc,
                0,
            )
            record += (
                " | overtrain countdown: g="
                f"{remaining_epochs_gen},d={remaining_epochs_disc} |"
                f" avg-gen-loss={avg_global_gen_loss:.3f} | avg-"
                f"disc-loss={avg_global_disc_loss:.3f}"
            )

            if remaining_epochs_disc == 0 or remaining_epochs_gen == 0:
                record += (
                    f"\nOvertraining detected at epoch {epoch} with average"
                    f" generator loss {avg_global_gen_loss:.3f} and discriminator loss"
                    f" {avg_global_disc_loss:.3f}"
                )
                done = True
        print(record)

        # Save weights, checkpoints and reference inference results
        # every N epochs
        if epoch % save_every_epoch == 0:
            with (
                torch.amp.autocast(
                    device_type="cuda", enabled=use_amp, dtype=train_dtype
                ),
                torch.no_grad(),
            ):
                if hasattr(net_g, "module"):
                    o, *_ = net_g.module.infer(*reference)
                else:
                    o, *_ = net_g.infer(*reference)
            audio_dict = {f"gen/audio_{global_step:07d}": o[0, :, :]}
            summarize(
                writer=writer,
                global_step=global_step,
                images=image_dict,
                scalars=scalar_dict,
                audios=audio_dict,
                audio_sample_rate=config.data.sample_rate,
            )
            checkpoint_idxs.append(2333333)
            if not save_only_latest:
                checkpoint_idxs.append(epoch)

            if custom_save_every_weights:
                model_add.append(
                    os.path.join(experiment_dir, f"{model_name}_{epoch}.pth"),
                )
        else:
            summarize(
                writer=writer,
                global_step=global_step,
                images=image_dict,
                scalars=scalar_dict,
            )
        for idx in checkpoint_idxs:
            save_checkpoint(
                net_g,
                optim_g,
                config.train.learning_rate,
                epoch,
                lowest_g_value,
                consecutive_increases_gen,
                os.path.join(experiment_dir, f"G_{idx}.pth"),
                scaler,
            )
            save_checkpoint(
                net_d,
                optim_d,
                config.train.learning_rate,
                epoch,
                lowest_d_value,
                consecutive_increases_disc,
                os.path.join(experiment_dir, f"D_{idx}.pth"),
                scaler,
            )
        if model_add:
            ckpt = (
                net_g.module.state_dict()
                if hasattr(net_g, "module")
                else net_g.state_dict()
            )
            for m in model_add:
                extract_model(
                    ckpt=ckpt,
                    sr=sample_rate,
                    name=model_name,
                    model_path=m,
                    epoch=epoch,
                    step=global_step,
                    hps=config,
                    overtrain_info=overtrain_info,
                    vocoder=vocoder,
                )
        # Check completion
        if epoch >= custom_total_epoch:
            lowest_g_value_rounded = float(lowest_g_value["value"])
            lowest_g_value_rounded = round(lowest_g_value_rounded, 3)
            print(
                f"Training has been successfully completed with {epoch} epoch(s),"
                f" {global_step} step(s) and {round(avg_global_gen_loss, 3)} average"
                " generator loss.",
            )
            print(
                f"Lowest average generator loss: {lowest_g_value_rounded} at epoch"
                f" {lowest_g_value['epoch']}",
            )

            done = True
        with torch.no_grad():
            torch.cuda.empty_cache()
        if done:
            cleanup_training_processes(experiment_dir)
            os._exit(0)


def cleanup_training_processes(experiment_dir) -> None:
    dist.destroy_process_group()
    pid_file_path = os.path.join(experiment_dir, "config.json")
    with pathlib.Path(pid_file_path).open() as pid_file:
        pid_data = json.load(pid_file)
    with pathlib.Path(pid_file_path).open("w") as pid_file:
        pid_data.pop("process_pids", None)
        json.dump(pid_data, pid_file, indent=4)
