"""
train_ternary_codec.py

Training script for a very small, low-bitrate neural audio codec that uses
BitNet-style ternary weight quantization ({-1, 0, +1} + a per-layer scale)
to keep the model tiny (weights compress to ~1.58 bits each), inspired by
Stability AI's stable-codec (encoder/decoder + vector quantized latent
bottleneck) but drastically shrunk.

Architecture
------------
Waveform -> [TernaryConv1d encoder, strided downsampling] -> latent frames
         -> [Residual Vector Quantizer (RVQ), small codebooks]
         -> [TernaryConvTranspose1d decoder, upsampling] -> waveform

Ternary quantization (BitNet b1.58 style)
------------------------------------------
Each ternary layer keeps full-precision "latent" weights for the optimizer,
but the forward pass uses a quantized version:
    scale  = mean(|W|)                       (per-tensor)
    W_tern = scale * round(clip(W / (scale + eps), -1, 1))
Gradients pass straight through the rounding op (straight-through estimator).
Activations are quantized to 8-bit per-token in the same spirit as BitNet,
which is optional here and controlled by `quantize_activations`.

This script is self-contained: model, ternary layers, RVQ, losses, dataset,
and training loop. It expects a folder of audio files (wav/flac/mp3) and
trains a reconstruction + commitment loss codec. For best perceptual
quality you would normally add adversarial (GAN) + STFT losses and a
discriminator -- a simple multi-resolution STFT loss is included, and a
lightweight discriminator hook is provided but off by default to keep the
whole thing small and fast to train.

Usage
-----
python train_ternary_codec.py --data_dir /path/to/wavs --epochs 100 \
    --sample_rate 16000 --batch_size 16 --out_dir ./runs/ternary_codec

Dependencies: torch, torchaudio, numpy
"""

import argparse
import glob
import math
import os
import random
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

try:
    import torchaudio
except ImportError as e:
    raise ImportError(
        "torchaudio is required. Install with `pip install torchaudio --break-system-packages`"
    ) from e

# torchaudio's newer default backend (torchcodec) requires FFmpeg's shared
# libraries to be discoverable on the system, which frequently isn't the case
# out of the box on Windows and raises DLL load errors. We instead load audio
# via the `soundfile` backend (pure Python via libsndfile, `pip install
# soundfile`) whenever possible. See `_load_audio` below.
try:
    import soundfile as _sf
    _HAS_SOUNDFILE = True
except ImportError:
    _HAS_SOUNDFILE = False


def _load_audio(path: str) -> Tuple[torch.Tensor, int]:
    """Load an audio file as (channels, samples) float tensor + sample rate.

    Prefers the `soundfile` backend to avoid torchcodec/FFmpeg DLL issues;
    falls back to torchaudio's default loader if soundfile isn't installed
    or fails to load a given file.
    """
    if _HAS_SOUNDFILE:
        try:
            data, sr = _sf.read(path, dtype="float32", always_2d=True)  # (samples, channels)
            wav = torch.from_numpy(data.T)  # (channels, samples)
            return wav, sr
        except Exception:
            pass  # fall through to torchaudio
    return torchaudio.load(path)


# --------------------------------------------------------------------------
# Ternary (BitNet 1.58-bit) layers
# --------------------------------------------------------------------------

def _ternary_quantize(w: torch.Tensor, eps: float = 1e-5) -> Tuple[torch.Tensor, torch.Tensor]:
    """Per-tensor absmean ternary quantization with straight-through estimator.

    Returns (w_quant, scale). w_quant has the same shape as w, values in
    {-scale, 0, +scale}. Gradients flow straight through to `w`.
    """
    scale = w.abs().mean().clamp(min=eps)
    w_scaled = w / scale
    w_rounded = torch.clamp(torch.round(w_scaled), -1, 1)
    # straight-through estimator: forward uses rounded*scale, backward acts as identity
    w_quant = w_scaled + (w_rounded - w_scaled).detach()
    w_quant = w_quant * scale
    return w_quant, scale


def _activation_quantize(x: torch.Tensor, bits: int = 8, eps: float = 1e-5) -> torch.Tensor:
    """Per-token (last-dim-collapsed) symmetric quantization, straight-through."""
    qmax = 2 ** (bits - 1) - 1
    scale = x.abs().amax(dim=-1, keepdim=True).clamp(min=eps) / qmax
    x_scaled = x / scale
    x_rounded = torch.clamp(torch.round(x_scaled), -qmax, qmax)
    x_quant = x_scaled + (x_rounded - x_scaled).detach()
    return x_quant * scale


class TernaryConv1d(nn.Module):
    """Conv1d whose weights are ternary-quantized on every forward pass.

    Full-precision "shadow" weights are what the optimizer actually updates;
    only the quantized version is used for the actual convolution, with a
    straight-through gradient estimator connecting the two.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        bias: bool = True,
        quantize_activations: bool = False,
    ):
        super().__init__()
        self.stride = stride
        self.padding = padding
        self.quantize_activations = quantize_activations
        self.weight = nn.Parameter(
            torch.empty(out_channels, in_channels, kernel_size)
        )
        nn.init.kaiming_normal_(self.weight, nonlinearity="linear")
        self.bias = nn.Parameter(torch.zeros(out_channels)) if bias else None
        # LayerNorm-style pre-activation scaling helps ternary nets train (BitNet uses this)
        self.norm = nn.GroupNorm(1, in_channels, affine=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        if self.quantize_activations:
            x = _activation_quantize(x.transpose(1, 2)).transpose(1, 2)
        w_q, _ = _ternary_quantize(self.weight)
        return F.conv1d(x, w_q, self.bias, stride=self.stride, padding=self.padding)


class TernaryConvTranspose1d(nn.Module):
    """Transposed Conv1d with ternary-quantized weights (decoder upsampling)."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        output_padding: int = 0,
        bias: bool = True,
        quantize_activations: bool = False,
    ):
        super().__init__()
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.quantize_activations = quantize_activations
        self.weight = nn.Parameter(
            torch.empty(in_channels, out_channels, kernel_size)
        )
        nn.init.kaiming_normal_(self.weight, nonlinearity="linear")
        self.bias = nn.Parameter(torch.zeros(out_channels)) if bias else None
        self.norm = nn.GroupNorm(1, in_channels, affine=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        if self.quantize_activations:
            x = _activation_quantize(x.transpose(1, 2)).transpose(1, 2)
        w_q, _ = _ternary_quantize(self.weight)
        return F.conv_transpose1d(
            x, w_q, self.bias, stride=self.stride, padding=self.padding,
            output_padding=self.output_padding,
        )


# --------------------------------------------------------------------------
# Residual Vector Quantizer
# --------------------------------------------------------------------------

class VectorQuantizer(nn.Module):
    """Single-stage VQ with EMA codebook updates (van den Oord et al.)."""

    def __init__(self, dim: int, codebook_size: int, decay: float = 0.99, eps: float = 1e-5):
        super().__init__()
        self.dim = dim
        self.codebook_size = codebook_size
        self.decay = decay
        self.eps = eps

        embed = torch.randn(codebook_size, dim) * 0.01
        self.register_buffer("embed", embed)
        self.register_buffer("cluster_size", torch.zeros(codebook_size))
        self.register_buffer("embed_avg", embed.clone())

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # x: (B, T, D)
        flat = x.reshape(-1, self.dim)
        dist = (
            flat.pow(2).sum(1, keepdim=True)
            - 2 * flat @ self.embed.t()
            + self.embed.pow(2).sum(1)
        )
        idx = dist.argmin(dim=1)
        onehot = F.one_hot(idx, self.codebook_size).type(flat.dtype)
        quantized = onehot @ self.embed
        quantized = quantized.view_as(x)

        if self.training:
            with torch.no_grad():
                cluster_size = onehot.sum(0)
                embed_sum = onehot.t() @ flat
                self.cluster_size.mul_(self.decay).add_(cluster_size, alpha=1 - self.decay)
                self.embed_avg.mul_(self.decay).add_(embed_sum, alpha=1 - self.decay)
                n = self.cluster_size.sum()
                cluster_size = (
                    (self.cluster_size + self.eps) / (n + self.codebook_size * self.eps) * n
                )
                embed_normalized = self.embed_avg / cluster_size.unsqueeze(1)
                self.embed.copy_(embed_normalized)

        commitment_loss = F.mse_loss(x, quantized.detach())
        # straight-through estimator so gradients reach the encoder
        quantized_st = x + (quantized - x).detach()
        return quantized_st, commitment_loss, idx.view(x.shape[0], x.shape[1])


class ResidualVQ(nn.Module):
    """Stack of VQ stages, each quantizing the residual of the previous one."""

    def __init__(self, dim: int, num_stages: int, codebook_size: int):
        super().__init__()
        self.stages = nn.ModuleList(
            [VectorQuantizer(dim, codebook_size) for _ in range(num_stages)]
        )

    def forward(self, x: torch.Tensor):
        residual = x
        quantized_out = torch.zeros_like(x)
        total_loss = 0.0
        all_indices = []
        for stage in self.stages:
            quantized, loss, idx = stage(residual)
            residual = residual - quantized
            quantized_out = quantized_out + quantized
            total_loss = total_loss + loss
            all_indices.append(idx)
        indices = torch.stack(all_indices, dim=-1)  # (B, T, num_stages)
        return quantized_out, total_loss / len(self.stages), indices


# --------------------------------------------------------------------------
# Encoder / Decoder
# --------------------------------------------------------------------------

@dataclass
class CodecConfig:
    sample_rate: int = 16000
    channels: int = 1
    base_channels: int = 24          # keep small -> small model
    latent_dim: int = 32
    strides: List[int] = field(default_factory=lambda: [2, 4, 5, 5])  # total downsample = 200x
    kernel_mult: int = 2             # kernel = stride * kernel_mult (typical codec setting)
    num_rvq_stages: int = 4
    codebook_size: int = 256         # 8 bits per stage -> 4 stages = 32 bits/frame
    quantize_activations: bool = False

    @property
    def hop_length(self) -> int:
        h = 1
        for s in self.strides:
            h *= s
        return h

    @property
    def bitrate_kbps(self) -> float:
        frames_per_sec = self.sample_rate / self.hop_length
        bits_per_frame = self.num_rvq_stages * math.log2(self.codebook_size)
        return frames_per_sec * bits_per_frame / 1000.0


class Encoder(nn.Module):
    def __init__(self, cfg: CodecConfig):
        super().__init__()
        c = cfg.base_channels
        layers = [TernaryConv1d(cfg.channels, c, kernel_size=7, padding=3,
                                 quantize_activations=cfg.quantize_activations)]
        in_c = c
        for i, s in enumerate(cfg.strides):
            out_c = min(c * (2 ** (i + 1)), 256)
            k = s * cfg.kernel_mult + 1
            layers.append(nn.GELU())
            layers.append(
                TernaryConv1d(in_c, out_c, kernel_size=k, stride=s, padding=k // 2,
                              quantize_activations=cfg.quantize_activations)
            )
            in_c = out_c
        layers.append(nn.GELU())
        layers.append(
            TernaryConv1d(in_c, cfg.latent_dim, kernel_size=3, padding=1,
                          quantize_activations=cfg.quantize_activations)
        )
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)  # (B, latent_dim, T')


class TernaryUpsampleConv1d(nn.Module):
    """Upsample (nearest) + ternary conv, used instead of ConvTranspose1d.

    Plain transposed convolutions are notorious for producing periodic
    "checkerboard" amplitude-modulation artifacts when kernel_size isn't an
    exact multiple of stride (which is unavoidable here for odd strides like
    5). Upsample-then-convolve avoids the uneven overlap-add entirely: the
    nearest-neighbor upsample just repeats samples (no learned weighting to
    go uneven), and the following conv is a normal, evenly-supported
    convolution. This is a standard fix used in several lightweight vocoders
    for exactly this artifact.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        quantize_activations: bool = False,
    ):
        super().__init__()
        self.stride = stride
        # kernel_size here should be odd so "same"-style padding is exact
        if kernel_size % 2 == 0:
            kernel_size += 1
        self.conv = TernaryConv1d(
            in_channels, out_channels, kernel_size=kernel_size,
            stride=1, padding=kernel_size // 2,
            quantize_activations=quantize_activations,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=self.stride, mode="nearest")
        return self.conv(x)


class Decoder(nn.Module):
    def __init__(self, cfg: CodecConfig):
        super().__init__()
        c = cfg.base_channels
        channel_seq = [min(c * (2 ** (i + 1)), 256) for i in range(len(cfg.strides))]
        channel_seq = list(reversed(channel_seq))
        layers = [TernaryConv1d(cfg.latent_dim, channel_seq[0], kernel_size=3, padding=1,
                                 quantize_activations=cfg.quantize_activations)]
        in_c = channel_seq[0]
        rev_strides = list(reversed(cfg.strides))
        for i, s in enumerate(rev_strides):
            out_c = channel_seq[i + 1] if i + 1 < len(channel_seq) else c
            k = s * cfg.kernel_mult + 1
            layers.append(nn.GELU())
            layers.append(
                TernaryUpsampleConv1d(
                    in_c, out_c, kernel_size=k, stride=s,
                    quantize_activations=cfg.quantize_activations,
                )
            )
            in_c = out_c
        layers.append(nn.GELU())
        layers.append(
            TernaryConv1d(in_c, cfg.channels, kernel_size=7, padding=3,
                          quantize_activations=cfg.quantize_activations)
        )
        layers.append(nn.Tanh())
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TernaryCodec(nn.Module):
    def __init__(self, cfg: CodecConfig):
        super().__init__()
        self.cfg = cfg
        self.encoder = Encoder(cfg)
        self.rvq = ResidualVQ(cfg.latent_dim, cfg.num_rvq_stages, cfg.codebook_size)
        self.decoder = Decoder(cfg)

    def forward(self, wav: torch.Tensor):
        # wav: (B, 1, T)
        z = self.encoder(wav)                     # (B, D, T')
        z_t = z.transpose(1, 2)                    # (B, T', D)
        q, vq_loss, indices = self.rvq(z_t)
        q = q.transpose(1, 2)                       # (B, D, T')
        recon = self.decoder(q)
        # pad/crop to match input length
        recon = _match_length(recon, wav.shape[-1])
        return recon, vq_loss, indices

    @torch.no_grad()
    def num_ternary_params(self) -> int:
        n = 0
        for m in self.modules():
            if isinstance(m, (TernaryConv1d, TernaryConvTranspose1d)):
                n += m.weight.numel()
        return n


def _match_length(x: torch.Tensor, target_len: int) -> torch.Tensor:
    cur_len = x.shape[-1]
    if cur_len == target_len:
        return x
    if cur_len > target_len:
        return x[..., :target_len]
    return F.pad(x, (0, target_len - cur_len))


# --------------------------------------------------------------------------
# Losses
# --------------------------------------------------------------------------

class MultiResolutionSTFTLoss(nn.Module):
    """Sum of L1 losses on log-magnitude spectrograms at several FFT sizes."""

    def __init__(self, fft_sizes=(512, 1024, 2048), hop_ratio=0.25, win_ratio=1.0):
        super().__init__()
        self.fft_sizes = fft_sizes
        self.hop_ratio = hop_ratio
        self.win_ratio = win_ratio

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # x, y: (B, 1, T)
        loss = 0.0
        x = x.squeeze(1)
        y = y.squeeze(1)
        for n_fft in self.fft_sizes:
            hop = int(n_fft * self.hop_ratio)
            win = int(n_fft * self.win_ratio)
            window = torch.hann_window(win, device=x.device)
            X = torch.stft(x, n_fft=n_fft, hop_length=hop, win_length=win,
                            window=window, return_complex=True)
            Y = torch.stft(y, n_fft=n_fft, hop_length=hop, win_length=win,
                            window=window, return_complex=True)
            X_mag = torch.log(X.abs().clamp(min=1e-5))
            Y_mag = torch.log(Y.abs().clamp(min=1e-5))
            loss = loss + F.l1_loss(X_mag, Y_mag)
        return loss / len(self.fft_sizes)


# --------------------------------------------------------------------------
# Dataset
# --------------------------------------------------------------------------

class AudioFolderDataset(Dataset):
    """Loads random fixed-length crops from every audio file in a directory."""

    AUDIO_EXTS = (".wav", ".flac", ".mp3", ".ogg")

    def __init__(self, data_dir: str, sample_rate: int, segment_seconds: float = 1.0):
        self.files = []
        for ext in self.AUDIO_EXTS:
            self.files.extend(glob.glob(os.path.join(data_dir, "**", f"*{ext}"), recursive=True))
        if not self.files:
            raise RuntimeError(f"No audio files found under {data_dir}")
        self.sample_rate = sample_rate
        self.segment_len = int(segment_seconds * sample_rate)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        wav, sr = _load_audio(path)
        wav = _preprocess_wav(wav, sr, self.sample_rate, self.segment_len)
        return wav


class LibriTTSDataset(Dataset):
    """Loads LibriTTS waveforms directly from disk (no transcripts needed).

    Expects the standard LibriTTS layout: root/LibriTTS/<subset>/<speaker>/
    <chapter>/<utterance>.wav (this is what torchaudio's own downloader/
    extractor produces, and what you get if you download+extract LibriTTS
    yourself). We scan for .wav files ourselves and decode them via the
    soundfile-preferring `_load_audio` helper, sidestepping torchaudio's
    LIBRITTS dataset class (which internally uses the torchcodec backend and
    requires FFmpeg's shared libraries to be on the system).

    If the expected files aren't found and `download=True`, falls back to
    torchaudio.datasets.LIBRITTS just to perform the download+extract step
    (decoding still goes through `_load_audio`).
    """

    SUBSETS = (
        "dev-clean", "dev-other", "test-clean", "test-other",
        "train-clean-100", "train-clean-360", "train-other-500",
    )

    def __init__(
        self,
        root: str,
        sample_rate: int,
        segment_seconds: float = 1.0,
        subset: str = "train-clean-100",
        download: bool = True,
    ):
        if subset not in self.SUBSETS:
            raise ValueError(f"Unknown LibriTTS subset '{subset}', expected one of {self.SUBSETS}")
        os.makedirs(root, exist_ok=True)

        subset_dir = os.path.join(root, "LibriTTS", subset)
        self.files = glob.glob(os.path.join(subset_dir, "**", "*.wav"), recursive=True)

        if not self.files:
            if not download:
                raise RuntimeError(
                    f"No .wav files found under {subset_dir}. Either the subset hasn't been "
                    f"downloaded/extracted there yet, or the folder layout doesn't match "
                    f"root/LibriTTS/<subset>/... . Pass --download (without --no_download) "
                    f"to fetch it automatically, or point --data_dir at the right root."
                )
            # Use torchaudio's dataset only to trigger the download+extract;
            # we still read files ourselves afterward to avoid its loader.
            torchaudio.datasets.LIBRITTS(root=root, url=subset, download=True)
            self.files = glob.glob(os.path.join(subset_dir, "**", "*.wav"), recursive=True)
            if not self.files:
                raise RuntimeError(
                    f"Downloaded LibriTTS but still found no .wav files under {subset_dir}. "
                    f"Check the extracted folder layout."
                )

        self.sample_rate = sample_rate
        self.segment_len = int(segment_seconds * sample_rate)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        wav, sr = _load_audio(self.files[idx])
        return _preprocess_wav(wav, sr, self.sample_rate, self.segment_len)


def _preprocess_wav(wav: torch.Tensor, sr: int, target_sr: int, segment_len: int) -> torch.Tensor:
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr)
    if wav.shape[-1] < segment_len:
        wav = F.pad(wav, (0, segment_len - wav.shape[-1]))
    else:
        start = random.randint(0, wav.shape[-1] - segment_len)
        wav = wav[:, start:start + segment_len]
    # peak-normalize to avoid clipping / dead silence blowing up losses
    peak = wav.abs().max().clamp(min=1e-5)
    wav = wav / peak * 0.95
    return wav


# --------------------------------------------------------------------------
# Training loop
# --------------------------------------------------------------------------

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)

    strides = [int(s) for s in args.strides.split(",")]
    cfg = CodecConfig(
        sample_rate=args.sample_rate,
        base_channels=args.base_channels,
        latent_dim=args.latent_dim,
        strides=strides,
        num_rvq_stages=args.num_rvq_stages,
        codebook_size=args.codebook_size,
        quantize_activations=args.quantize_activations,
    )
    model = TernaryCodec(cfg).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    n_ternary = model.num_ternary_params()
    approx_bits = n_ternary * math.log2(3) + (n_params - n_ternary) * 32
    print(f"Total params:            {n_params:,}")
    print(f"Ternary-quantized params: {n_ternary:,} ({n_ternary / n_params:.1%})")
    print(f"Approx model size:        {approx_bits / 8 / 1e6:.2f} MB "
          f"(vs {n_params * 4 / 1e6:.2f} MB full fp32)")
    print(f"Hop length:               {cfg.hop_length} samples "
          f"({cfg.sample_rate / cfg.hop_length:.1f} frames/sec)")
    print(f"Target bitrate:           {cfg.bitrate_kbps:.2f} kbps")

    if args.dataset == "libritts":
        dataset = LibriTTSDataset(
            root=args.data_dir,
            sample_rate=args.sample_rate,
            segment_seconds=args.segment_seconds,
            subset=args.libritts_subset,
            download=args.download,
        )
    else:
        dataset = AudioFolderDataset(args.data_dir, args.sample_rate, args.segment_seconds)
    print(f"Dataset: {args.dataset} ({len(dataset)} utterances)")
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, drop_last=True, pin_memory=True,
    )

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs * len(loader))
    stft_loss_fn = MultiResolutionSTFTLoss().to(device)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    start_epoch = 0
    step = 0
    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        # Our checkpoints store a CodecConfig dataclass alongside the tensors, so
        # PyTorch 2.6+'s default weights_only=True load will refuse it. This is
        # our own trusted checkpoint format (saved by this same script), so it's
        # safe to opt out of weights_only restriction here.
        try:
            torch.serialization.add_safe_globals([CodecConfig])
        except AttributeError:
            pass  # older torch versions don't have add_safe_globals; fall back below
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        if "optimizer" in ckpt:
            opt.load_state_dict(ckpt["optimizer"])
        if "scheduler" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler"])
        if "scaler" in ckpt and args.amp:
            scaler.load_state_dict(ckpt["scaler"])
        start_epoch = ckpt.get("epoch", -1) + 1
        step = ckpt.get("step", 0)
        print(f"Resumed at epoch {start_epoch}, step {step}")
        if ckpt.get("cfg") is not None and vars(ckpt["cfg"]) != vars(cfg):
            print("WARNING: checkpoint's CodecConfig differs from the current CLI args' "
                  "config. Continuing with the CURRENT config/model architecture; make "
                  "sure --strides/--num_rvq_stages/--codebook_size/etc. match the run "
                  "you're resuming, or shapes may not load correctly.")

    for epoch in range(start_epoch, args.epochs):
        model.train()
        running = {"recon": 0.0, "stft": 0.0, "vq": 0.0, "total": 0.0}
        for wav in loader:
            wav = wav.to(device, non_blocking=True)

            with torch.cuda.amp.autocast(enabled=args.amp):
                recon, vq_loss, _ = model(wav)
                recon_loss = F.l1_loss(recon, wav)
                stft_loss = stft_loss_fn(recon.float(), wav.float())
                total_loss = (
                    args.recon_weight * recon_loss
                    + args.stft_weight * stft_loss
                    + args.vq_weight * vq_loss
                )

            opt.zero_grad(set_to_none=True)
            scaler.scale(total_loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(opt)
            scaler.update()
            scheduler.step()

            running["recon"] += recon_loss.item()
            running["stft"] += stft_loss.item()
            running["vq"] += vq_loss.item()
            running["total"] += total_loss.item()
            step += 1

            if step % args.log_every == 0:
                n = args.log_every
                print(
                    f"epoch {epoch} step {step} "
                    f"total={running['total']/n:.4f} "
                    f"recon={running['recon']/n:.4f} "
                    f"stft={running['stft']/n:.4f} "
                    f"vq={running['vq']/n:.4f} "
                    f"lr={scheduler.get_last_lr()[0]:.2e}"
                )
                running = {k: 0.0 for k in running}

        ckpt_path = os.path.join(args.out_dir, f"codec_epoch{epoch:04d}.pt")
        torch.save({
            "model": model.state_dict(),
            "optimizer": opt.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "cfg": cfg,
            "epoch": epoch,
            "step": step,
        }, ckpt_path)
        print(f"Saved checkpoint: {ckpt_path}")

    final_path = os.path.join(args.out_dir, "codec_final.pt")
    torch.save({
        "model": model.state_dict(),
        "optimizer": opt.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "cfg": cfg,
        "epoch": args.epochs - 1,
        "step": step,
    }, final_path)
    print(f"Training complete. Final model: {final_path}")


def parse_args():
    p = argparse.ArgumentParser(description="Train a tiny ternary-quantized neural audio codec")
    p.add_argument("--data_dir", type=str, required=True,
                    help="For --dataset folder: directory of audio files. "
                         "For --dataset libritts: root dir to store/read LibriTTS in "
                         "(torchaudio will create a LibriTTS/ subfolder there).")
    p.add_argument("--out_dir", type=str, default="./runs/ternary_codec")
    p.add_argument("--resume", type=str, default=None,
                    help="Path to a checkpoint (.pt) to resume training from. Loads model, "
                         "optimizer, scheduler and AMP scaler state, and continues from the "
                         "saved epoch/step. Model/codec config (--strides, --num_rvq_stages, "
                         "--codebook_size, --base_channels, --latent_dim, --sample_rate) must "
                         "match the run that produced the checkpoint.")
    p.add_argument("--dataset", type=str, default="folder", choices=["folder", "libritts"],
                    help="'folder' = any directory of wav/flac/mp3/ogg files, "
                         "'libritts' = torchaudio's LibriTTS corpus")
    p.add_argument("--libritts_subset", type=str, default="train-clean-100",
                    choices=list(LibriTTSDataset.SUBSETS),
                    help="Which LibriTTS split to use (only for --dataset libritts)")
    p.add_argument("--download", action="store_true", default=True,
                    help="Auto-download the LibriTTS subset if not already present")
    p.add_argument("--no_download", dest="download", action="store_false",
                    help="Disable auto-download; assume LibriTTS is already on disk")
    p.add_argument("--sample_rate", type=int, default=24000,
                    help="LibriTTS is natively 24kHz; audio is resampled to this rate. "
                         "Use 16000 to keep the codec/bitrate settings from the folder-dataset examples.")
    p.add_argument("--segment_seconds", type=float, default=1.0)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--log_every", type=int, default=50)
    p.add_argument("--amp", action="store_true", help="Use mixed precision training")

    # model size / bitrate knobs
    p.add_argument("--base_channels", type=int, default=24)
    p.add_argument("--latent_dim", type=int, default=32)
    p.add_argument("--strides", type=str, default="2,4,5,5",
                    help="Comma-separated encoder/decoder strides. Their product is the "
                         "hop length (samples per latent frame); frame_rate = sample_rate / hop. "
                         "Lower frame_rate = lower bitrate. E.g. '4,5,5,8' -> hop=800.")
    p.add_argument("--num_rvq_stages", type=int, default=4)
    p.add_argument("--codebook_size", type=int, default=256)
    p.add_argument("--quantize_activations", action="store_true",
                    help="Also apply 8-bit activation quantization (full BitNet-style)")

    # loss weights
    p.add_argument("--recon_weight", type=float, default=1.0)
    p.add_argument("--stft_weight", type=float, default=1.0)
    p.add_argument("--vq_weight", type=float, default=0.25)

    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    train(args)