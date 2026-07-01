"""Convert *_mask.png cable masks to COCO instance segmentation JSON."""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
from pycocotools import mask as mask_utils


IMAGE_DIR = Path("/home/hea4sgh/fiftyone/cable_images")
IMAGE_EXTS = {
    ".jpg", ".jpeg", ".jpe", ".jfif", ".png", ".webp", ".bmp",
    ".gif", ".tif", ".tiff", ".avif", ".heic", ".heif", ".jp2",
    ".j2k", ".tga", ".ppm", ".pgm", ".pbm", ".pnm",
}


def is_source_image(path):
    return (
        path.suffix.lower() in IMAGE_EXTS
        and not path.stem.endswith("_mask")
        and "_sam3_" not in path.stem
    )

# find images and corresponding masks
def mask_path(image_path):
    return image_path.with_name(f"{image_path.stem}_mask.png")


def encode_mask(mask):
    rle = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
    rle["counts"] = rle["counts"].decode("ascii")
    return rle


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-dir", type=Path, default=IMAGE_DIR)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--category", default="cable")
    args = parser.parse_args()

    image_dir = args.image_dir.expanduser().resolve()
    out_path = args.out or image_dir / "annotations_coco.json"
    images, annotations = [], []
    ann_id = 1

    source_images = sorted(p for p in image_dir.iterdir() if is_source_image(p))
    for image_id, image_path in enumerate(source_images, start=1):
        image = Image.open(image_path).convert("RGB")
        width, height = image.size
        images.append(
            {
                "id": image_id,
                "file_name": image_path.name,
                "width": width,
                "height": height,
            }
        )

        path = mask_path(image_path)
        if not path.exists():
            print(f"skip missing mask: {path.name}")
            continue

        mask_image = Image.open(path).convert("L")
        if mask_image.size != image.size:
            mask_image = mask_image.resize(image.size, Image.Resampling.NEAREST)

        mask = np.asarray(mask_image) > 0
        if not mask.any():
            print(f"skip empty mask: {path.name}")
            continue

        rle = encode_mask(mask)
        annotations.append(
            {
                "id": ann_id,
                "image_id": image_id,
                "category_id": 1,
                "segmentation": rle,
                "area": int(mask_utils.area(rle)),
                "bbox": [float(x) for x in mask_utils.toBbox(rle)],
                "iscrowd": 0,
            }
        )
        ann_id += 1

    coco = {
        "images": images,
        "annotations": annotations,
        "categories": [{"id": 1, "name": args.category, "supercategory": "object"}],
    }
    out_path.write_text(json.dumps(coco, indent=2), encoding="utf-8")
    print(f"wrote {out_path}")
    print(f"images={len(images)} annotations={len(annotations)}")


if __name__ == "__main__":
    main()