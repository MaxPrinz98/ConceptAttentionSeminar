import os
import argparse
from pathlib import Path
from PIL import Image
import numpy as np
import matplotlib.cm as cm


def process_directory(directory_path, threshold=0.5, force=False):
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
        # Avoid processing already generated files
        if (
            heatmap_file.name.startswith("upscaled_")
            or heatmap_file.name.startswith("mask_")
            or heatmap_file.name.startswith("masked_image_")
        ):
            continue

        concept_name = heatmap_file.stem.replace("heatmap_", "")

        # Check if already processed (skip unless --force)
        masked_img_path = dir_path / f"masked_image_{concept_name}.png"
        if masked_img_path.exists() and not force:
            print(f"  Skipping concept: {concept_name} (already processed)")
            continue

        print(f"  Processing concept: {concept_name}")

        try:
            # Load heatmap as grayscale
            heatmap = Image.open(heatmap_file).convert("L")

            # 1. Upscale heatmap to img_size (1024x1024)
            upscaled_heatmap = heatmap.resize(img_size, resample=Image.BICUBIC)

            # --- Save color-mapped upscaled heatmap ---
            heatmap_np = np.array(upscaled_heatmap).astype(float) / 255.0
            colormap = cm.get_cmap("inferno")
            colored = colormap(heatmap_np)  # Returns RGBA float [0,1]
            colored_rgb = (colored[:, :, :3] * 255).astype(np.uint8)
            upscaled_path = dir_path / f"upscaled_heatmap_{concept_name}.png"
            Image.fromarray(colored_rgb).save(upscaled_path)

            # 2. Generate Mask via relative threshold
            # threshold * max_value: pixels above this are considered "active"
            max_val = np.max(heatmap_np)
            cutoff = threshold * max_val if max_val > 0 else 0
            mask_np = (heatmap_np > cutoff).astype(np.uint8) * 255
            mask = Image.fromarray(mask_np)
            mask_path = dir_path / f"mask_{concept_name}.png"
            mask.save(mask_path)

            # 3. Create Masked Image (original image × binary mask)
            img_np = np.array(img)
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
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force reprocessing of all heatmaps, even if already done.",
    )

    args = parser.parse_args()

    if args.single_dir:
        print(f"Processing single directory: {args.single_dir}")
        process_directory(args.single_dir, args.threshold, force=args.force)
    else:
        root_path = Path(args.root)
        print(f"Scanning {root_path} for set_* directories...")

        for root, dirs, files in os.walk(root_path):
            if Path(root).name.startswith("set_"):
                print(f"Found directory: {root}")
                process_directory(root, args.threshold, force=args.force)


if __name__ == "__main__":
    main()
