import datetime
import glob
import json
import logging
import os
import shutil
import sys
from collections import deque
from random import randint, shuffle
from time import time as ttime

import numpy as np
from tqdm import tqdm

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# Zluda hijack
import ultimate_rvc.rvc.lib.zluda
from ultimate_rvc.common import TRAINING_MODELS_DIR
from ultimate_rvc.rvc.common import RVC_TRAINING_MODELS_DIR
from ultimate_rvc.rvc.lib.algorithm import commons
from ultimate_rvc.rvc.train.losses import (
    discriminator_loss,
    envelope_loss,
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
    save_checkpoint,
    summarize,
)

logging.getLogger("torch").setLevel(logging.ERROR)
logger = logging.getLogger(__name__)

torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True
torch.multiprocessing.set_start_method("spawn", force=True)

randomized = True
optimizer = "RAdam"  # "AdamW"
global_step = 0
lowest_g_value = {"value": float("inf"), "epoch": 0}
lowest_d_value = {"value": float("inf"), "epoch": 0}
consecutive_increases_gen = 0
consecutive_increases_disc = 0

avg_losses = {
    "gen_loss_queue": deque(maxlen=10),
    "disc_loss_queue": deque(maxlen=10),
    "disc_loss_50": deque(maxlen=50),
    "env_loss_50": deque(maxlen=50),
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
        return f"time={current_time} | speed={elapsed_time_str}"


def verify_checkpoint_shapes(checkpoint_path: str, model: torch.nn.Module) -> None:
    """
    Verify that the checkpoint and model have the same
    architecture.
    """
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    checkpoint_state_dict = checkpoint["model"]
    try:
        if hasattr(model, "module"):
            model_state_dict = model.module.load_state_dict(checkpoint_state_dict)
        else:
            model_state_dict = model.load_state_dict(checkpoint_state_dict)
    except RuntimeError:
        logger.error(  # noqa: TRY400
            "The parameters of the pretrain model such as the sample rate or"
            " architecture do not match the selected model.",
        )
        sys.exit(1)
    else:
        del checkpoint
        del checkpoint_state_dict
        del model_state_dict


def main(
    model_name: str,
    sample_rate: int,
    version: str,
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
    gpus: set[int],
) -> None:
    """Start the training process."""
    experiment_dir = os.path.join(TRAINING_MODELS_DIR, model_name)
    config_save_path = os.path.join(experiment_dir, "config.json")

    # Use a Manager to create a shared list
    manager = mp.Manager()
    global_gen_loss = manager.list([0] * total_epoch)
    global_disc_loss = manager.list([0] * total_epoch)

    with open(config_save_path) as f:
        config = json.load(f)
    config = HParams(**config)
    config.data.training_files = os.path.join(experiment_dir, "filelist.txt")

    # Set up distributed training environment for master node.
    # master node is localhost because we are running on a single local
    # machine. master port is randomly selected
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(randint(20000, 55555))
    # Check sample rate
    wavs = glob.glob(
        os.path.join(experiment_dir, "sliced_audios", "*.wav"),
    )
    if wavs:
        _, sr = load_wav_to_torch(wavs[0])
        if sr != sample_rate:
            logger.error(
                "Error: Pretrained model sample rate (%d Hz) does not match dataset"
                " audio sample rate (%d Hz).",
                sample_rate,
                sr,
            )
            os._exit(1)
    else:
        logger.warning("No wav file found.")

    if torch.cuda.is_available():
        device = torch.device("cuda")
        n_gpus = len(gpus)
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        gpus = {0}
        n_gpus = 1
    else:
        device = torch.device("cpu")
        gpus = {0}
        n_gpus = 1
        logger.warning("Training with CPU, this will take a long time.")

    def start() -> None:
        """Start the training process with multi-GPU support or CPU."""
        children = []
        pid_data = {"process_pids": []}
        with open(config_save_path) as pid_file:
            try:
                existing_data = json.load(pid_file)
                pid_data.update(existing_data)
            except json.JSONDecodeError:
                pass
        with open(config_save_path, "w") as pid_file:
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
                        version,
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
                    ),
                )
                children.append(subproc)
                subproc.start()
                pid_data["process_pids"].append(subproc.pid)
            json.dump(pid_data, pid_file, indent=4)

        for i in range(n_gpus):
            children[i].join()

    if cleanup:
        logger.info("Removing files from the prior training attempt...")

        # Clean up unnecessary files
        for entry in os.scandir(os.path.join(TRAINING_MODELS_DIR, model_name)):
            if entry.is_file():
                _, file_extension = os.path.splitext(entry.name)
                if file_extension in {".0", ".pth", ".index"}:
                    os.remove(entry.path)
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
    version,
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
    global global_step, optimizer

    if rank == 0:
        writer_eval = SummaryWriter(log_dir=os.path.join(experiment_dir, "eval"))
    else:
        writer_eval = None

    # Initialize distributed training environment for child node.
    dist.init_process_group(
        backend="gloo",
        init_method="env://?use_libuv=0",
        world_size=n_gpus if device.type == "cuda" else 1,
        rank=rank if device.type == "cuda" else 0,
    )

    torch.manual_seed(config.train.seed)

    if torch.cuda.is_available():
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
        is_half=config.train.fp16_run and device.type == "cuda",
        sr=sample_rate,
        vocoder=vocoder,
        checkpointing=checkpointing,
        randomized=randomized,
    )

    net_d = MultiPeriodDiscriminator(
        version,
        config.model.use_spectral_norm,
        checkpointing=checkpointing,
    )

    if torch.cuda.is_available():
        net_g = net_g.cuda(device_id)
        net_d = net_d.cuda(device_id)
    else:
        net_g.to(device)
        net_d.to(device)

    if optimizer == "AdamW":
        optimizer = torch.optim.AdamW
    elif optimizer == "RAdam":
        optimizer = torch.optim.RAdam

    optim_g = optimizer(
        net_g.parameters(),
        config.train.learning_rate,
        betas=config.train.betas,
        eps=config.train.eps,
    )
    optim_d = optimizer(
        net_d.parameters(),
        config.train.learning_rate,
        betas=config.train.betas,
        eps=config.train.eps,
    )

    fn_mel_loss = MultiScaleMelSpectrogramLoss(sample_rate=sample_rate)

    # Wrap models with DDP for multi-gpu processing
    if n_gpus > 1 and device.type == "cuda":
        net_g = DDP(net_g, device_ids=[device_id])
        net_d = DDP(net_d, device_ids=[device_id])

    # Load checkpoint if available
    try:
        logger.info("Starting training...")
        _, _, _, epoch_str = load_checkpoint(
            latest_checkpoint_path(experiment_dir, "D_*.pth"),
            net_d,
            optim_d,
        )
        _, _, _, epoch_str = load_checkpoint(
            latest_checkpoint_path(experiment_dir, "G_*.pth"),
            net_g,
            optim_g,
        )
        epoch_str += 1
        global_step = (epoch_str - 1) * len(train_loader)

    except:
        epoch_str = 1
        global_step = 0
        if pretrain_g not in {"", "None"}:
            if rank == 0:
                verify_checkpoint_shapes(pretrain_g, net_g)
                logger.info("Loaded pretrained (G) '%s'", pretrain_g)
            if hasattr(net_g, "module"):
                net_g.module.load_state_dict(
                    torch.load(pretrain_g, map_location="cpu", weights_only=False)[
                        "model"
                    ],
                )
            else:
                net_g.load_state_dict(
                    torch.load(pretrain_g, map_location="cpu", weights_only=False)[
                        "model"
                    ],
                )

        if pretrain_d not in {"", "None"}:
            if rank == 0:
                verify_checkpoint_shapes(pretrain_d, net_d)
                logger.info("Loaded pretrained (D) '%s'", pretrain_d)
            if hasattr(net_d, "module"):
                net_d.module.load_state_dict(
                    torch.load(pretrain_d, map_location="cpu", weights_only=False)[
                        "model"
                    ],
                )
            else:
                net_d.load_state_dict(
                    torch.load(pretrain_d, map_location="cpu", weights_only=False)[
                        "model"
                    ],
                )

    # Initialize schedulers and scaler
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

    scaler = GradScaler("cuda", enabled=config.train.fp16_run and device.type == "cuda")

    cache = []
    # get the first sample as reference for tensorboard evaluation
    # custom reference temporarily disabled
    if True == False and os.path.isfile(
        os.path.join(RVC_TRAINING_MODELS_DIR, "reference", f"ref{sample_rate}.wav"),
    ):
        phone = np.load(
            os.path.join(
                RVC_TRAINING_MODELS_DIR,
                "reference",
                f"ref{sample_rate}_feats.npy",
            ),
        )
        # expanding x2 to match pitch size
        phone = np.repeat(phone, 2, axis=0)
        phone = torch.FloatTensor(phone).unsqueeze(0).to(device)
        phone_lengths = torch.LongTensor(phone.size(0)).to(device)
        pitch = np.load(
            os.path.join(
                RVC_TRAINING_MODELS_DIR,
                "reference",
                f"ref{sample_rate}_f0c.npy",
            ),
        )
        # removed last frame to match features
        pitch = torch.LongTensor(pitch[:-1]).unsqueeze(0).to(device)
        pitchf = np.load(
            os.path.join(
                RVC_TRAINING_MODELS_DIR,
                "reference",
                f"ref{sample_rate}_f0f.npy",
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
        for info in train_loader:
            phone, phone_lengths, pitch, pitchf, _, _, _, _, sid = info
            if device.type == "cuda":
                reference = (
                    phone.cuda(device_id, non_blocking=True),
                    phone_lengths.cuda(device_id, non_blocking=True),
                    pitch.cuda(device_id, non_blocking=True),
                    pitchf.cuda(device_id, non_blocking=True),
                    sid.cuda(device_id, non_blocking=True),
                )
            else:
                reference = (
                    phone.to(device),
                    phone_lengths.to(device),
                    pitch.to(device),
                    pitchf.to(device),
                    sid.to(device),
                )
            break

    for epoch in range(epoch_str, custom_total_epoch + 1):
        train_and_evaluate(
            rank,
            epoch,
            config,
            [net_g, net_d],
            [optim_g, optim_d],
            [scheduler_g, scheduler_d],
            scaler,
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
            version,
            vocoder,
            save_every_epoch,
            save_only_latest,
            overtraining_detector,
            overtraining_threshold,
            cache_data_in_gpu,
            global_gen_loss,
            global_disc_loss,
        )


def train_and_evaluate(
    rank,
    epoch,
    config,
    nets,
    optims,
    schedulers,
    scaler,
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
    version,
    vocoder,
    save_every_epoch,
    save_only_latest,
    overtraining_detector,
    overtraining_threshold,
    cache_data_in_gpu,
    global_gen_loss,
    global_disc_loss,
) -> None:
    """Train and evaluates the model for one epoch."""
    global global_step, lowest_g_value, lowest_d_value, consecutive_increases_gen, consecutive_increases_disc

    epoch_disc_sum = 0.0
    epoch_gen_sum = 0.0

    model_add = []
    checkpoint_idxs = [2333333]
    done = False

    net_g, net_d = nets
    optim_g, optim_d = optims
    scheduler_g, scheduler_d = schedulers
    skip_g_scheduler, skip_d_scheduler = False, False
    train_loader = loaders[0] if loaders is not None else None
    if writers is not None:
        writer = writers[0]

    train_loader.batch_sampler.set_epoch(epoch)

    net_g.train()
    net_d.train()

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

            # Forward pass
            use_amp = config.train.fp16_run and device.type == "cuda"
            with autocast("cuda", enabled=use_amp):
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
                y_d_hat_r, y_d_hat_g, _, _ = net_d(wave, y_hat.detach())
                with autocast("cuda", enabled=False):
                    # if vocoder == "HiFi-GAN":
                    #    loss_disc, _, _ = discriminator_loss(y_d_hat_r, y_d_hat_g)
                    # else:
                    #    loss_disc, _, _ = discriminator_loss_scaled(y_d_hat_r, y_d_hat_g)
                    loss_disc, _, _ = discriminator_loss(y_d_hat_r, y_d_hat_g)
            # Discriminator backward and update
            epoch_disc_sum += loss_disc
            global_disc_loss[epoch - 1] += loss_disc.item()
            optim_d.zero_grad()
            scaler.scale(loss_disc).backward()
            scaler.unscale_(optim_d)
            grad_norm_d = torch.nn.utils.clip_grad_norm_(
                net_d.parameters(),
                max_norm=1000.0,
            )
            scaler.step(optim_d)
            d_scale = scaler.get_scale()
            scaler.update()
            skip_d_scheduler = d_scale > scaler.get_scale()
            # if not math.isfinite(grad_norm_d):
            #    print("\nWarning: grad_norm_d is NaN or Inf")

            # Generator backward and update
            with autocast("cuda", enabled=use_amp):
                _, y_d_hat_g, fmap_r, fmap_g = net_d(wave, y_hat)
                with autocast("cuda", enabled=False):
                    loss_mel = fn_mel_loss(wave, y_hat) * config.train.c_mel / 3.0
                    loss_env = envelope_loss(wave, y_hat)
                    loss_kl = (
                        kl_loss(z_p, logs_q, m_p, logs_p, z_mask) * config.train.c_kl
                    )
                    loss_fm = feature_loss(fmap_r, fmap_g)
                    # if vocoder == "HiFi-GAN":
                    # 	loss_gen, _ = generator_loss(y_d_hat_g)
                    # else:
                    # 	loss_gen, _ = generator_loss_scaled(y_d_hat_g)
                    loss_gen, _ = generator_loss(y_d_hat_g)
                    loss_gen_all = loss_gen + loss_fm + loss_mel + loss_kl + loss_env
            epoch_gen_sum += loss_gen_all
            global_gen_loss[epoch - 1] += loss_gen_all.item()
            optim_g.zero_grad()
            scaler.scale(loss_gen_all).backward()
            scaler.unscale_(optim_g)
            grad_norm_g = torch.nn.utils.clip_grad_norm_(
                net_g.parameters(),
                max_norm=1000.0,
            )
            scaler.step(optim_g)
            g_scale = scaler.get_scale()
            scaler.update()
            skip_g_scheduler = g_scale > scaler.get_scale()
            # if not math.isfinite(grad_norm_g):
            #    print("\n Warning: grad_norm_g is NaN or Inf")

            global_step += 1

            # queue for rolling losses over 50 steps
            avg_losses["disc_loss_50"].append(loss_disc.detach())
            avg_losses["env_loss_50"].append(loss_env.detach())
            avg_losses["fm_loss_50"].append(loss_fm.detach())
            avg_losses["kl_loss_50"].append(loss_kl.detach())
            avg_losses["mel_loss_50"].append(loss_mel.detach())
            avg_losses["gen_loss_50"].append(loss_gen_all.detach())

            if rank == 0 and global_step % 50 == 0:
                # logging rolling averages
                scalar_dict = {
                    "loss_avg_50/d/total": torch.mean(
                        torch.stack(list(avg_losses["disc_loss_50"])),
                    ),
                    "loss_avg_50/g/env": torch.mean(
                        torch.stack(list(avg_losses["env_loss_50"])),
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

    if not skip_d_scheduler:
        scheduler_d.step()
    if not skip_g_scheduler:
        scheduler_g.step()

    with torch.no_grad():
        torch.cuda.empty_cache()
    # Logging and checkpointing
    if rank == 0:
        avg_losses["disc_loss_queue"].append(epoch_disc_sum.item() / len(train_loader))
        avg_losses["gen_loss_queue"].append(epoch_gen_sum.item() / len(train_loader))
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
        with autocast("cuda", enabled=False):
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
            if use_amp:
                y_hat_mel = y_hat_mel.half()

        lr = optim_g.param_groups[0]["lr"]

        scalar_dict = {
            "loss/g/total": loss_gen_all,
            "loss/d/total": loss_disc,
            "learning_rate": lr,
            "grad/norm_d": grad_norm_d.item(),
            "grad/norm_g": grad_norm_g.item(),
            "loss/g/fm": loss_fm,
            "loss/g/mel": loss_mel,
            "loss/g/kl": loss_kl,
            "loss/g/env": loss_env,
            "loss_avg_epoch/disc": np.mean(avg_losses["disc_loss_queue"]),
            "loss_avg_epoch/gen": np.mean(avg_losses["gen_loss_queue"]),
        }
        # commented out
        # scalar_dict.update({f"loss/g/{i}": v for i, v in enumerate(losses_gen)})
        # scalar_dict.update({f"loss/d_r/{i}": v for i, v in enumerate(losses_disc_r)})
        # scalar_dict.update({f"loss/d_g/{i}": v for i, v in enumerate(losses_disc_g)})

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

            remaining_epochs_gen = overtraining_threshold - consecutive_increases_gen
            remaining_epochs_disc = (
                overtraining_threshold * 2 - consecutive_increases_disc
            )
            record += (
                " | overtrain countdown: g="
                f"{remaining_epochs_gen},d={remaining_epochs_disc} |"
                f" avg-gen-loss={avg_global_gen_loss:.3f} | avg-"
                f"disc-loss={avg_global_disc_loss:.3f}"
            )

            if (
                consecutive_increases_gen == overtraining_threshold
                or consecutive_increases_disc == overtraining_threshold * 2
            ):
                record += (
                    f"\nOvertraining detected at epoch {epoch} with average"
                    f" generator loss {avg_global_gen_loss:.3f} and discriminator loss"
                    f" {avg_global_disc_loss:.3f}"
                )
                done = True
        print(record)

        # Save weights, checkpoints and reference inference resulsts
        # every N epochs
        if epoch % save_every_epoch == 0:
            with torch.no_grad():
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
                os.path.join(experiment_dir, f"G_{idx}.pth"),
            )
            save_checkpoint(
                net_d,
                optim_d,
                config.train.learning_rate,
                epoch,
                os.path.join(experiment_dir, f"D_{idx}.pth"),
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
                    pitch_guidance=True,
                    name=model_name,
                    model_dir=m,
                    epoch=epoch,
                    step=global_step,
                    version=version,
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
            pid_file_path = os.path.join(experiment_dir, "config.json")
            with open(pid_file_path) as pid_file:
                pid_data = json.load(pid_file)
            with open(pid_file_path, "w") as pid_file:
                pid_data.pop("process_pids", None)
                json.dump(pid_data, pid_file, indent=4)
            os._exit(2333333)
