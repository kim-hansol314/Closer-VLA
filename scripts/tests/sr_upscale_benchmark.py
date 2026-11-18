"""
Does super-resolution (Real-ESRGAN) help the magnify crop, and what does it cost?

Uses LIBERO RLDS shards (e.g., openvla/modified_libero_rlds tfrecords + features.json), reproduces the training
image (resize to 224 + jpeg), detects the target with the same TargetLocator as the annotation script, and
takes the magnify crop window from `crop_to_bboxes`. Frames are sampled shortly before the gripper releases,
i.e., where magnify is gated on.

1. Quality with a reference: the 224 frame is downsampled 2x, the same window is cropped from the low-res frame
   and upscaled 2x by each method, then compared to the crop of the 224 frame (PSNR / SSIM / LPIPS, and cosine
   similarity of SigLIP features after resizing both to 224).
2. Latency: each method producing a 224x224 crop, batch 1 (inference) and batch 32 (training), per device.
3. Visual grid of the real setting (crop of the 224 frame upscaled to 224).

    python scripts/tests/sr_upscale_benchmark.py \
        --tfrecords libero_spatial-00000.tfrecord libero_goal-00000.tfrecord \
        --features_json features.json --weights_dir weights --out_dir sr_results
"""

import argparse
import json
import os
import sys
import time
from collections import defaultdict

import cv2
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

sys.path.append(os.path.expanduser("./"))

from src.agent.target_locator import TargetLocator
from src.data.dlimp.utils import resize_image
from src.model.vla.processing import crop_to_bboxes

tf.config.set_visible_devices([], "GPU")

# fidelity (PSNR) oriented models trained on bicubic downsampling, and GAN models for real-world degradations
SR_MODELS = {
    # classical
    "swinir_m_x2": "001_classicalSR_DF2K_s64w8_SwinIR-M_x2.pth",
    "swinir_m_x4": "001_classicalSR_DF2K_s64w8_SwinIR-M_x4.pth",
    "swinir_s_x2": "002_lightweightSR_DIV2K_s64w8_SwinIR-S_x2.pth",
    "swin2sr_x2": "Swin2SR_ClassicalSR_X2_64.pth",
    "hat_l_x4": "HAT-L_SRx4_ImageNet-pretrain.pth",
    # GAN
    "esrgan_x4plus": "RealESRGAN_x4plus.pth",
    "esrgan_x2plus": "RealESRGAN_x2plus.pth",
    "general_x4v3": "realesr-general-x4v3.pth",
    "bsrgan_x4": "BSRGAN.pth",
    "bsrgan_x2": "BSRGANx2.pth",
    "swinir_gan_x4": "003_realSR_BSRGAN_DFO_s64w8_SwinIR-M_x4_GAN.pth",
}
INTERP_METHODS = ["bilinear", "bicubic", "lanczos"]
LR_KERNELS = {"area": cv2.INTER_AREA, "bicubic": cv2.INTER_CUBIC}


# ---------- data ----------#


def training_frame(image_256: np.ndarray) -> np.ndarray:
    """same as ResizeAndJpegEncode: resize to 224, stored as jpeg"""
    resized = tf.cast(resize_image(tf.constant(image_256), (224, 224)), tf.uint8)
    return tf.io.decode_jpeg(tf.io.encode_jpeg(resized)).numpy()


def load_episodes(tfrecords, features_json, frames_before_release):
    features = tfds.features.FeatureConnector.from_json(json.load(open(features_json)))
    episodes = []
    for path in tfrecords:
        for raw in tf.data.TFRecordDataset(path):
            steps = next(iter(features.deserialize_example(raw)["steps"].batch(100000)))
            images = steps["observation"]["image"].numpy()
            gripper_closed = steps["action"][:, -1].numpy() > 0  # raw libero: +1 close, -1 open
            releases = np.where(gripper_closed[:-1] & ~gripper_closed[1:])[0] + 1
            t = (
                max(int(releases[-1]) - frames_before_release, 0)
                if len(releases)
                else len(images) // 2
            )
            episodes.append(
                {
                    "suite": os.path.basename(path).split("-")[0],
                    "instruction": steps["language_instruction"][0].numpy().decode(),
                    "first": training_frame(images[0]),
                    "frame": training_frame(images[t]),
                    "t": t,
                    "has_release": bool(len(releases)),
                }
            )
    return episodes


def crop_window(bbox, size=224):
    """the window magnify would crop, in pixels (x0, y0, side)"""
    dummy = torch.zeros(1, 3, size, size, dtype=torch.uint8)
    _, window = crop_to_bboxes(dummy, torch.tensor([bbox]), out_size=8)
    x0, y0, x1, _ = (window[0] * size).round().int().tolist()
    return x0, y0, x1 - x0


# ---------- upscaling ----------#


class Upscalers:
    def __init__(self, weights_dir, device, half=False, models=None):
        from spandrel import ImageModelDescriptor, ModelLoader

        self.device = device
        self.dtype = torch.float16 if half else torch.float32
        self.models = {}
        for name, file in SR_MODELS.items():
            if models is not None and name not in models:
                continue
            model = ModelLoader().load_from_file(os.path.join(weights_dir, file))
            assert isinstance(model, ImageModelDescriptor)
            model.to(device).eval()
            if half:
                if not model.supports_half:
                    print(f"{name} does not support fp16, skipped")
                    continue
                model.half()
            # call the descriptor, not .model: it pads to the size requirements (x2plus needs multiples of 4)
            self.models[name] = model

    @property
    def names(self):
        return INTERP_METHODS + list(self.models)

    @torch.no_grad()
    def __call__(self, name, x: torch.Tensor, size: int) -> torch.Tensor:
        """x: [B, 3, h, w] float in [0, 1] on device -> [B, 3, size, size]"""
        if name in ("bilinear", "bicubic"):
            return F.interpolate(x, size=(size, size), mode=name, align_corners=False, antialias=True).clamp(0, 1)
        if name == "lanczos":
            out = [
                np.asarray(Image.fromarray(to_uint8(img)).resize((size, size), Image.LANCZOS))
                for img in x
            ]
            return torch.stack([from_uint8(o) for o in out]).to(x.device)
        model = self.models[name]
        y = model(x.to(self.dtype)).float().clamp(0, 1)
        if y.shape[-1] != size:  # e.g., x4 output larger than needed
            y = F.interpolate(y, size=(size, size), mode="bilinear", align_corners=False, antialias=True)
        return y.clamp(0, 1)


def to_uint8(x: torch.Tensor) -> np.ndarray:
    return (x.detach().float().cpu().clamp(0, 1).permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)


def from_uint8(x: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(x).permute(2, 0, 1).float() / 255


# ---------- quality ----------#


def evaluate_quality(samples, upscalers, siglip, out_dir, lr_kernel="area"):
    import lpips
    from skimage.metrics import peak_signal_noise_ratio, structural_similarity

    device = upscalers.device
    lpips_fn = lpips.LPIPS(net="alex", verbose=False).to(device)
    metrics = defaultdict(lambda: defaultdict(list))
    for sample in samples:
        frame = sample["frame"]
        x0, y0, side = sample["window"]
        # even-aligned so the low-res window is exact
        x0, y0, side = x0 - x0 % 2, y0 - y0 % 2, side - side % 2
        gt = frame[y0 : y0 + side, x0 : x0 + side]
        low = cv2.resize(frame, (112, 112), interpolation=LR_KERNELS[lr_kernel])
        low_crop = low[y0 // 2 : (y0 + side) // 2, x0 // 2 : (x0 + side) // 2]
        gt_t = from_uint8(gt)[None].to(device)
        gt_feat = siglip(gt_t)
        for name in upscalers.names:
            out = upscalers(name, from_uint8(low_crop)[None].to(device), side)
            out_np = to_uint8(out[0])
            m = metrics[name]
            m["psnr"].append(peak_signal_noise_ratio(gt, out_np, data_range=255))
            m["ssim"].append(structural_similarity(gt, out_np, channel_axis=2, data_range=255))
            with torch.no_grad():
                m["lpips"].append(float(lpips_fn(out * 2 - 1, gt_t * 2 - 1)))
            pooled, patches = siglip(out)
            m["siglip_pooled_cos"].append(float(F.cosine_similarity(pooled, gt_feat[0]).mean()))
            m["siglip_patch_cos"].append(float(F.cosine_similarity(patches, gt_feat[1], dim=-1).mean()))
    summary = {
        name: {k: float(np.mean(v)) for k, v in m.items()} for name, m in metrics.items()
    }
    json.dump(summary, open(os.path.join(out_dir, f"quality_{lr_kernel}.json"), "w"), indent=2)
    return summary


class SiglipFeatures:
    """SigLIP base as a proxy for the PaliGemma vision tower (so400m), features after resizing to 224"""

    def __init__(self, device, model_id="google/siglip-base-patch16-224"):
        from transformers import SiglipVisionModel

        self.model = SiglipVisionModel.from_pretrained(model_id).to(device).eval()

    @torch.no_grad()
    def __call__(self, x):
        x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False, antialias=True)
        out = self.model(pixel_values=(x - 0.5) / 0.5)
        return out.pooler_output, out.last_hidden_state


# ---------- latency ----------#


def sync(device):
    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()


def time_method(upscalers, name, batch, in_side, repeats):
    device = upscalers.device
    x = torch.rand(batch, 3, in_side, in_side, device=device)
    for _ in range(3):
        upscalers(name, x, 224)
    sync(device)
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        upscalers(name, x, 224)
        sync(device)
        times.append(time.perf_counter() - start)
    return float(np.median(times)) * 1000


def benchmark_latency(weights_dir, devices, typical_side, repeats, models=None):
    results = []
    for device, half in devices:
        upscalers = Upscalers(weights_dir, device, half=half, models=models)
        # SR models get a fixed input size so batches can be stacked: x4 from 56, x2 from 112
        inputs = {name: typical_side for name in INTERP_METHODS}
        inputs.update({name: 224 // model.scale for name, model in upscalers.models.items()})
        for name in upscalers.names:
            if half and name in INTERP_METHODS:
                continue
            for batch in (1, 32):
                if name == "lanczos" and batch == 32 and device != "cpu":
                    continue
                try:
                    ms = time_method(upscalers, name, batch, inputs[name], repeats)
                except Exception as e:  # e.g., ops unsupported on mps
                    print(f"{device} {name} batch {batch} failed: {e!r}"[:300])
                    continue
                results.append({"device": device + (" fp16" if half else ""), "method": name,
                                "input": inputs[name], "batch": batch, "ms_per_batch": ms,
                                "ms_per_crop": ms / batch})
                print(f"{results[-1]['device']:9s} {name:14s} in {inputs[name]:3d} batch {batch:2d}: "
                      f"{ms:8.2f} ms/batch  {ms / batch:7.2f} ms/crop")
    return results


# ---------- visuals ----------#


def save_grid(samples, upscalers, path, max_rows=8):
    names = upscalers.names
    cell, label_h = 160, 18
    rows = samples[:max_rows]
    grid = Image.new("RGB", (cell * (len(names) + 1), (cell + label_h) * len(rows)), "white")
    draw = ImageDraw.Draw(grid)
    for r, sample in enumerate(rows):
        y = r * (cell + label_h)
        x0, y0, side = sample["window"]
        full = Image.fromarray(sample["frame"]).copy()
        ImageDraw.Draw(full).rectangle([x0, y0, x0 + side, y0 + side], outline=(255, 0, 0), width=2)
        grid.paste(full.resize((cell, cell)), (0, y + label_h))
        draw.text((2, y + 2), f"{sample['phrase']} ({side}px)", fill=(0, 0, 0))
        crop = from_uint8(sample["frame"][y0 : y0 + side, x0 : x0 + side])[None].to(upscalers.device)
        for c, name in enumerate(names, start=1):
            out = to_uint8(upscalers(name, crop, 224)[0])
            grid.paste(Image.fromarray(out).resize((cell, cell), Image.NEAREST), (c * cell, y + label_h))
            if r == 0:
                draw.text((c * cell + 2, y + 2), name, fill=(0, 0, 0))
    grid.save(path)


def save_detections(samples, path, cols=8):
    cell = 160
    rows = (len(samples) + cols - 1) // cols
    grid = Image.new("RGB", (cell * cols, cell * rows), "white")
    for i, sample in enumerate(samples):
        img = Image.fromarray(sample["first"]).copy()
        draw = ImageDraw.Draw(img)
        if sample.get("bbox"):
            x0, y0, x1, y1 = [v * 224 for v in sample["bbox"]]
            draw.rectangle([x0, y0, x1, y1], outline=(255, 0, 0), width=2)
        draw.text((2, 2), f"{sample.get('phrase')} {sample.get('score', 0):.2f}", fill=(255, 255, 0))
        grid.paste(img.resize((cell, cell)), ((i % cols) * cell, (i // cols) * cell))
    grid.save(path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tfrecords", nargs="+", required=True)
    parser.add_argument("--features_json", required=True)
    parser.add_argument("--weights_dir", required=True)
    parser.add_argument("--out_dir", default="sr_results")
    parser.add_argument("--frames_before_release", type=int, default=5)
    parser.add_argument("--detector_device", default="cpu")
    parser.add_argument("--quality_device", default="cpu")
    parser.add_argument("--latency_devices", nargs="+", default=["cpu", "mps", "mps:fp16"])
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--models", nargs="+", default=list(SR_MODELS), choices=list(SR_MODELS))
    parser.add_argument("--lr_kernels", nargs="+", default=["area", "bicubic"], choices=list(LR_KERNELS))
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    t_start = time.time()
    episodes = load_episodes(args.tfrecords, args.features_json, args.frames_before_release)
    print(f"Loaded {len(episodes)} episodes in {time.time() - t_start:.1f}s")

    # detect the target on the first frame, as in the annotation script
    locator = TargetLocator(device=args.detector_device)
    det_times = []
    for episode in episodes:
        start = time.perf_counter()
        detection = locator.locate(episode["first"], episode["instruction"])
        det_times.append(time.perf_counter() - start)
        episode["phrase"] = locator.get_phrase(episode["instruction"])
        if detection is not None:
            episode.update(detection)
            episode["window"] = crop_window(detection["bbox"])
    save_detections(episodes, os.path.join(args.out_dir, "detections.png"))
    samples = [e for e in episodes if "window" in e]
    sides = np.array([s["window"][2] for s in samples])
    data_summary = {
        "episodes": len(episodes),
        "detected": len(samples),
        "with_release": int(sum(e["has_release"] for e in episodes)),
        "detector_ms_median": float(np.median(det_times)) * 1000,
        "crop_side_px": {"min": int(sides.min()), "median": float(np.median(sides)), "max": int(sides.max())},
        "upscale_factor_median": float(224 / np.median(sides)),
        "misses": [e["instruction"] for e in episodes if "window" not in e],
    }
    print(json.dumps(data_summary, indent=2))
    json.dump(data_summary, open(os.path.join(args.out_dir, "data.json"), "w"), indent=2)

    upscalers = Upscalers(args.weights_dir, args.quality_device, models=args.models)
    save_grid(samples[:: max(len(samples) // 8, 1)], upscalers, os.path.join(args.out_dir, "upscaled.png"))
    siglip = SiglipFeatures(args.quality_device)
    for lr_kernel in args.lr_kernels:
        quality = evaluate_quality(samples, upscalers, siglip, args.out_dir, lr_kernel=lr_kernel)
        print(f"\nQuality (2x from {lr_kernel} downsampling, vs. crop of the 224 frame):")
        for name, m in quality.items():
            print(f"  {name:14s} " + "  ".join(f"{k} {v:.4f}" for k, v in m.items()))

    devices = [(d.split(":")[0], d.endswith("fp16")) for d in args.latency_devices]
    latency = benchmark_latency(args.weights_dir, devices, int(np.median(sides)), args.repeats, args.models)
    json.dump(latency, open(os.path.join(args.out_dir, "latency.json"), "w"), indent=2)
