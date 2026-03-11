import os
import argparse
from pathlib import Path
from PIL import Image
import numpy as np


def process_directory(directory_path, threshold=0.5):
    """
    Processes a single directory containing image.png and heatmap_*.png files.
    """
    dir_path = Path(directory_path)
    image_file = dir_path / "image.png"

    if not image_file.exists():
        print(f"Skipping {dir_path}: image.png not found.")
        return

    # Load original image
    try:
        img = Image.open(image_file).convert("RGB")
        img_size = img.size  # (width, height), should be (1024, 1024)
    except Exception as e:
        print(f"Error loading image in {dir_path}: {e}")
        return

    # Find all heatmap files
    heatmap_files = list(dir_path.glob("heatmap_*.png"))

    for heatmap_file in heatmap_files:
        # Avoid processing already generated files (though glob pattern should be specific enough)
        if (
            heatmap_file.name.startswith("upscaled_")
            or heatmap_file.name.startswith("mask_")
            or heatmap_file.name.startswith("masked_image_")
        ):
            continue

        concept_name = heatmap_file.stem.replace("heatmap_", "")

        # Check if already processed
        masked_img_path = dir_path / f"masked_image_{concept_name}.png"
        if masked_img_path.exists():
            print(f"  Skipping concept: {concept_name} (already processed)")
            continue

        print(f"  Processing concept: {concept_name}")

        try:
            # Load heatmap
            heatmap = Image.open(heatmap_file).convert("L")  # Load as Grayscale

            # 1. Upscale heatmap to img_size (1024x1024)
            # Using BICUBIC for smooth upscaling
            upscaled_heatmap = heatmap.resize(img_size, resample=Image.BICUBIC)
            upscaled_path = dir_path / f"upscaled_heatmap_{concept_name}.png"
            upscaled_heatmap.save(upscaled_path)

            # 2. Generate Mask
            # Convert to numpy for thresholding
            heatmap_np = np.array(upscaled_heatmap).astype(float) / 255.0

            # Simple thresholding: values above threshold * max_val are kept
            # Or just absolute threshold if heatmap is normalized 0-1
            mask_np = (
                heatmap_np
                > (threshold * np.max(heatmap_np) if np.max(heatmap_np) > 0 else 0)
            ).astype(np.uint8) * 255
            mask = Image.fromarray(mask_np)
            mask_path = dir_path / f"mask_{concept_name}.png"
            mask.save(mask_path)

            # 3. Create Masked Image
            # Using the mask as an alpha channel or just multiplying
            # We want to keep only the highlighted pixels
            img_np = np.array(img)
            # Broadcast mask to 3 channels
            mask_3d = np.stack([mask_np] * 3, axis=-1) / 255.0
            masked_img_np = (img_np * mask_3d).astype(np.uint8)

            masked_img = Image.fromarray(masked_img_np)
            masked_img_path = dir_path / f"masked_image_{concept_name}.png"
            masked_img.save(masked_img_path)

        except Exception as e:
            print(f"    Error processing {heatmap_file.name}: {e}")


def main():
    parser = argparse.ArgumentParser(description="Upscale heatmaps and generate masks.")
    parser.add_argument(
        "--root",
        type=str,
        default="results/object_analysis",
        help="Root directory to search for set_* folders.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Threshold for mask generation (0.0 to 1.0).",
    )
    parser.add_argument(
        "--single-dir", type=str, help="Process only a specific directory."
    )

    args = parser.parse_args()

    if args.single_dir:
        print(f"Processing single directory: {args.single_dir}")
        process_directory(args.single_dir, args.threshold)
    else:
        root_path = Path(args.root)
        print(f"Scanning {root_path} for set_* directories...")

        # Walk through directories
        for root, dirs, files in os.walk(root_path):
            if Path(root).name.startswith("set_"):
                print(f"Found directory: {root}")
                process_directory(root, args.threshold)


if __name__ == "__main__":
    main()
