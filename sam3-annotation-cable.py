"""Auto-segment cable images with SAM3, edit in FiftyOne App, export PNG masks."""

import argparse
import os
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault(
    "CUDA_VISIBLE_DEVICES", "MIG-51c8054a-3ee8-524d-8985-081be630e2e1"
)

import numpy as np
import torch
from PIL import Image

import fiftyone as fo
import fiftyone.zoo as foz


IMAGE_DIR = Path("/home/hea4sgh/fiftyone/cable_images")
SAM3_DIR = Path(
    "/fs/scratch/rb_bd_dlp_rng_dl01_cr_PRS_employees/"
    "rai_models/hf-base-models/sam3"
)
ENTRYPOINT_ARGS = {
    "checkpoint_path": str(SAM3_DIR / "sam3.pt"),
    "bpe_path": str(SAM3_DIR / "bpe_simple_vocab_16e6.txt.gz"),
}

# CONCEPTS = [
#     "cable",
#     "charging cable",
#     "sync cable",
#     "USB cable",
#     "USB-C cable",
#     "Lightning cable",
#     "data cable",
#     "power cable",
#     "charger cord",
# ]

CONCEPTS = [
    "cable",
]


EXTS = {
    ".jpg", ".jpeg", ".jpe", ".jfif", ".png", ".webp", ".bmp",
    ".gif", ".tif", ".tiff", ".avif", ".heic", ".heif", ".jp2",
    ".j2k", ".tga", ".ppm", ".pgm", ".pbm", ".pnm",
}
LABEL = "cable"


def mask_path(image_path):
    return image_path.with_name(f"{image_path.stem}_mask.png")


def image_paths(image_dir):
    return sorted(
        p
        for p in image_dir.iterdir()
        if p.suffix.lower() in EXTS and "_sam3_" not in p.stem
    )


def save_mask(mask, path):
    Image.fromarray(mask.astype("uint8") * 255).save(path)


def detection_mask(detection, width, height):
    mask = detection.get_mask()
    if mask is None:
        return np.zeros((height, width), bool)

    mask = np.asarray(mask)
    if mask.ndim == 3:
        mask = mask[..., 0]
    mask = mask > 0
    if mask.shape == (height, width):
        return mask

    if not detection.bounding_box:
        return np.asarray(
            Image.fromarray(mask.astype("uint8") * 255).resize(
                (width, height), Image.Resampling.NEAREST
            )
        ) > 0

    x, y, w, h = detection.bounding_box
    x1, y1 = max(0, round(x * width)), max(0, round(y * height))
    x2 = min(width, round((x + w) * width))
    y2 = min(height, round((y + h) * height))
    if x2 <= x1 or y2 <= y1:
        return np.zeros((height, width), bool)

    if mask.shape != (y2 - y1, x2 - x1):
        mask = np.asarray(
            Image.fromarray(mask.astype("uint8") * 255).resize(
                (x2 - x1, y2 - y1), Image.Resampling.NEAREST
            )
        ) > 0

    output = np.zeros((height, width), bool)
    output[y1:y2, x1:x2] = mask[: y2 - y1, : x2 - x1]
    return output


def export_masks(dataset):
    for sample in dataset:
        image_path = Path(sample.filepath)
        image = Image.open(image_path).convert("RGB")
        mask = np.zeros((image.height, image.width), bool)
        labels = sample.get_field(LABEL)
        for detection in labels.detections if labels else []:
            mask |= detection_mask(detection, image.width, image.height)
        save_mask(mask, mask_path(image_path))
        print(f"saved {mask_path(image_path)}")


def new_dataset(paths, dataset_name):
    if fo.dataset_exists(dataset_name):
        fo.delete_dataset(dataset_name)
    dataset = fo.Dataset(dataset_name)
    dataset.add_samples([fo.Sample(filepath=str(p), media_type="image") for p in paths])
    dataset.persistent = True
    return dataset


def run_sam3(paths, dataset_name, mask_thresh):
    dataset = new_dataset(paths, dataset_name)
    model = foz.load_zoo_model(
        "segment-anything-3-image-torch",
        classes=CONCEPTS,
        operation_mode="concept",
        entrypoint_args=ENTRYPOINT_ARGS,
        output_processor_args={"mask_thresh": mask_thresh},
    )
    dataset.apply_model(model, label_field=LABEL)
    del model
    torch.cuda.empty_cache()
    return dataset


def dataset_from_existing_masks(paths, dataset_name):
    if fo.dataset_exists(dataset_name):
        return fo.load_dataset(dataset_name)

    dataset = new_dataset(paths, dataset_name)
    for sample in dataset:
        image_path = Path(sample.filepath)
        if mask_path(image_path).exists():
            mask = np.asarray(Image.open(mask_path(image_path)).convert("L")) > 0
            sample[LABEL] = fo.Detections(
                detections=[fo.Detection.from_mask(mask, label=LABEL)] if mask.any() else []
            )
            sample.save()
    return dataset


def edit_in_fiftyone(dataset, host, port):
    session = fo.launch_app(dataset, address=host, port=port, remote=True)
    print(f"FiftyOne App: http://{host}:{session.server_port}")
    print("Edit masks in the App. Press Ctrl+C here when finished to export PNG masks.")
    try:
        session.wait(-1)
    except KeyboardInterrupt:
        print("\nExporting edited masks...")
    finally:
        dataset.reload()
        export_masks(dataset)
        session.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-dir", type=Path, default=IMAGE_DIR)
    parser.add_argument("--dataset", default="sam3-cable-auto")
    parser.add_argument("--mask-thresh", type=float, default=0.5)
    parser.add_argument("--skip-inference", action="store_true")
    parser.add_argument("--no-app", "--no-editor", action="store_true")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5151)
    args = parser.parse_args()

    paths = image_paths(args.image_dir.expanduser().resolve())
    if not paths:
        raise SystemExit(f"no images found in {args.image_dir}")

    dataset = (
        dataset_from_existing_masks(paths, args.dataset)
        if args.skip_inference
        else run_sam3(paths, args.dataset, args.mask_thresh)
    )
    export_masks(dataset)
    if not args.no_app:
        edit_in_fiftyone(dataset, args.host, args.port)


if __name__ == "__main__":
    main()