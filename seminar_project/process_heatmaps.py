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

    # --- 4. Argmax Segmentation over all concepts ---
    # Only run argmax if we have concepts and force is True or segmentation doesn't exist
    seg_output_path = dir_path / "heatmap_segmentation.png"
    legend_output_path = dir_path / "segmentation_legend.json"

    if len(heatmap_files) > 0 and (force or not seg_output_path.exists()):
        print(f"  Generating Argmax Segmentation for {dir_path.name}")
        concept_names = [hf.stem.replace("heatmap_", "") for hf in heatmap_files]
        heatmaps = []

        try:
            for c_name in concept_names:
                upscaled_path = dir_path / f"upscaled_heatmap_{c_name}.png"
                if upscaled_path.exists():
                    img = Image.open(upscaled_path).convert("L")
                else:
                    # Fallback if upscaled doesn't exist for some reason
                    orig_path = dir_path / f"heatmap_{c_name}.png"
                    img = (
                        Image.open(orig_path)
                        .convert("L")
                        .resize(img_size, resample=Image.BICUBIC)
                    )

                heatmaps.append(np.array(img) / 255.0)

            # Add background layer (threshold: 0.1)
            bg_layer = np.full(img_size[::-1], 0.1)  # shape (H, W)
            stacked = np.stack([bg_layer] + heatmaps, axis=0)

            # Argmax segmentation
            segmentation_idx = np.argmax(stacked, axis=0)

            # Define colors (Background = Black)
            cmap = cm.get_cmap("tab10")
            colors = [(0, 0, 0)]  # Index 0 is background
            for i in range(len(concept_names)):
                r, g, b, _ = cmap((i % 10) / 10.0)
                colors.append((int(r * 255), int(g * 255), int(b * 255)))

            # Map indices to RGB colors
            seg_rgb = np.zeros((*img_size[::-1], 3), dtype=np.uint8)
            for i, color in enumerate(colors):
                seg_rgb[segmentation_idx == i] = color

            # Save segmentation image
            Image.fromarray(seg_rgb).save(seg_output_path)

            # Save Legend JSON
            legend_data = {"Background": "#000000"}
            for i, c_name in enumerate(concept_names):
                hex_color = "#{:02x}{:02x}{:02x}".format(*colors[i + 1])
                legend_data[c_name] = hex_color

            import json

            with open(legend_output_path, "w") as f:
                json.dump(legend_data, f, indent=4)

        except Exception as e:
            print(f"    Error generating segmentation for {dir_path.name}: {e}")


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
