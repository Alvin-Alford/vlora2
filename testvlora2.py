"""
test_ternary_codec.py

Load a trained checkpoint from train_ternary_codec.py, run it on one or more
audio files (encode -> quantize -> decode), and:
  - save the reconstructed audio next to the original so you can listen to it
  - print simple objective metrics (SI-SNR, L1, actual measured bitrate)
  - optionally evaluate over a whole directory and report averages

Usage
-----
# Single file
python test_ternary_codec.py --checkpoint runs/ternary_codec/codec_epoch0012.pt \
    --input path/to/clip.wav --out_dir ./test_outputs

# Whole folder (e.g. LibriTTS dev-clean, or a held-out folder of wavs)
python test_ternary_codec.py --checkpoint runs/ternary_codec/codec_epoch0012.pt \
    --input_dir C:\path\to\dev-clean --out_dir ./test_outputs --max_files 20
"""

import argparse
import glob
import math
import os

import torch
import torch.nn.functional as F

# Reuse everything from the training script so the model definition can never
# drift out of sync with what was actually trained.
from vlora2 import (
    TernaryCodec,
    CodecConfig,
    _load_audio,
    _preprocess_wav,
)

try:
    import soundfile as sf
except ImportError as e:
    raise ImportError("Install soundfile: pip install soundfile") from e


def si_snr(estimate: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> float:
    """Scale-invariant signal-to-noise ratio in dB. Higher is better."""
    estimate = estimate - estimate.mean()
    target = target - target.mean()
    proj = (torch.sum(estimate * target) / (torch.sum(target ** 2) + eps)) * target
    noise = estimate - proj
    ratio = torch.sum(proj ** 2) / (torch.sum(noise ** 2) + eps)
    return 10 * torch.log10(ratio + eps).item()


def load_model(checkpoint_path: str, device: torch.device) -> TernaryCodec:
    try:
        torch.serialization.add_safe_globals([CodecConfig])
    except AttributeError:
        pass
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg: CodecConfig = ckpt["cfg"]
    model = TernaryCodec(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"Loaded checkpoint from epoch {ckpt.get('epoch', '?')}, step {ckpt.get('step', '?')}")
    print(f"Config: sample_rate={cfg.sample_rate}, hop={cfg.hop_length}, "
          f"frames/sec={cfg.sample_rate / cfg.hop_length:.1f}, "
          f"rvq_stages={cfg.num_rvq_stages}, codebook_size={cfg.codebook_size}, "
          f"target_bitrate={cfg.bitrate_kbps:.3f} kbps")
    return model, cfg


@torch.no_grad()
def process_file(model: TernaryCodec, cfg: CodecConfig, path: str, device: torch.device,
                  bypass_vq: bool = False):
    wav, sr = _load_audio(path)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != cfg.sample_rate:
        import torchaudio
        wav = torchaudio.functional.resample(wav, sr, cfg.sample_rate)
    orig_len = wav.shape[-1]

    hop = cfg.hop_length
    pad = (hop - (orig_len % hop)) % hop
    wav_padded = F.pad(wav, (0, pad))
    wav_batched = wav_padded.unsqueeze(0).to(device)  # (1, 1, T)

    if bypass_vq:
        # Skip the RVQ bottleneck entirely: encoder -> decoder directly.
        # If the periodic glitch is STILL present, it's coming from the
        # conv/transposed-conv architecture, not codebook collapse.
        z = model.encoder(wav_batched)
        from vlora2 import _match_length
        recon = model.decoder(z)
        recon = _match_length(recon, wav_batched.shape[-1])
        vq_loss = torch.tensor(0.0)
        n_frames = z.shape[-1]
        indices = None
    else:
        recon, vq_loss, indices = model(wav_batched)
        n_frames = indices.shape[1]

    recon = recon.squeeze(0).cpu()[:, :orig_len]
    wav = wav[:, :orig_len]

    l1 = F.l1_loss(recon, wav).item()
    snr = si_snr(recon.squeeze(0), wav.squeeze(0))

    bits_per_frame = cfg.num_rvq_stages * math.log2(cfg.codebook_size)
    total_bits = n_frames * bits_per_frame
    duration_sec = orig_len / cfg.sample_rate
    measured_kbps = (total_bits / duration_sec) / 1000.0 if duration_sec > 0 else 0.0

    return recon, wav, {
        "l1": l1,
        "si_snr_db": snr,
        "vq_loss": vq_loss.item(),
        "duration_sec": duration_sec,
        "measured_kbps": measured_kbps,
    }


def main():
    p = argparse.ArgumentParser(description="Evaluate a trained ternary audio codec")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--input", type=str, default=None, help="Single audio file to test")
    p.add_argument("--input_dir", type=str, default=None,
                    help="Directory of audio files to test (recursive, wav/flac/mp3/ogg)")
    p.add_argument("--max_files", type=int, default=20,
                    help="Max files to process when using --input_dir")
    p.add_argument("--out_dir", type=str, default="./test_outputs")
    p.add_argument("--bypass_vq", action="store_true",
                    help="Skip RVQ quantization (encoder->decoder only). Use this to check "
                         "whether artifacts come from the conv architecture (still present "
                         "with bypass) or from codebook collapse/quantization (disappears "
                         "with bypass). Bitrate/measured_kbps is meaningless in this mode.")
    args = p.parse_args()

    if not args.input and not args.input_dir:
        raise ValueError("Pass either --input <file> or --input_dir <folder>")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)

    model, cfg = load_model(args.checkpoint, device)

    if args.input:
        files = [args.input]
    else:
        exts = ("*.wav", "*.flac", "*.mp3", "*.ogg")
        files = []
        for ext in exts:
            files.extend(glob.glob(os.path.join(args.input_dir, "**", ext), recursive=True))
        files = sorted(files)[: args.max_files]

    if not files:
        raise RuntimeError("No audio files found to test.")

    all_metrics = []
    for path in files:
        recon, orig, metrics = process_file(model, cfg, path, device, bypass_vq=args.bypass_vq)
        all_metrics.append(metrics)

        base = os.path.splitext(os.path.basename(path))[0]
        recon_path = os.path.join(args.out_dir, f"{base}_recon.wav")
        orig_path = os.path.join(args.out_dir, f"{base}_orig.wav")
        sf.write(recon_path, recon.squeeze(0).numpy(), cfg.sample_rate)
        sf.write(orig_path, orig.squeeze(0).numpy(), cfg.sample_rate)

        print(
            f"{base:30s} SI-SNR={metrics['si_snr_db']:6.2f} dB  "
            f"L1={metrics['l1']:.4f}  dur={metrics['duration_sec']:.2f}s  "
            f"measured_bitrate={metrics['measured_kbps']:.3f} kbps"
        )

    if len(all_metrics) > 1:
        avg_snr = sum(m["si_snr_db"] for m in all_metrics) / len(all_metrics)
        avg_l1 = sum(m["l1"] for m in all_metrics) / len(all_metrics)
        avg_kbps = sum(m["measured_kbps"] for m in all_metrics) / len(all_metrics)
        print("-" * 70)
        print(f"Averages over {len(all_metrics)} files: "
              f"SI-SNR={avg_snr:.2f} dB, L1={avg_l1:.4f}, bitrate={avg_kbps:.3f} kbps")

    print(f"\nReconstructed + original audio saved to: {args.out_dir}")
    print("Listen to the *_orig.wav vs *_recon.wav pairs to judge quality by ear.")


if __name__ == "__main__":
    main()