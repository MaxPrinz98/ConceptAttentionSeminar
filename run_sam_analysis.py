import os
import json
import argparse
import shutil
from pathlib import Path
import numpy as np
from PIL import Image
from ultralytics import SAM
from tqdm import tqdm


def calculate_iou(mask1, mask2):
    """Calculates Intersection over Union (IoU) between two binary masks."""
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    if union == 0:
        return 0.0
    return float(intersection / union)


def calculate_coverage(target_mask, reference_mask):
    """Calculates what percentage of the target mask is covered by the reference mask."""
    intersection = np.logical_and(target_mask, reference_mask).sum()
    target_sum = target_mask.sum()
    if target_sum == 0:
        return 0.0
    return float(intersection / target_sum)


def calculate_precision(target_mask, reference_mask):
    """Precision: what proportion of the reference mask is correct relative to the target mask."""
    intersection = np.logical_and(target_mask, reference_mask).sum()
    reference_sum = reference_mask.sum()
    if reference_sum == 0:
        return 0.0
    return float(intersection / reference_sum)


def calculate_recall(target_mask, reference_mask):
    """Recall: what proportion of the target mask is captured by the reference mask."""
    return calculate_coverage(target_mask, reference_mask)


def process_directory(set_path, model, device, force=False):
    """
    Processes a single 'set_*' directory:
    1. Runs SAM auto-segmentation.
    2. Matches concept masks to SAM segments greedily.
    3. Saves results in sam_analysis/ subdirectory.
    """
    set_path = Path(set_path)
    image_file = set_path / "image.png"
    metadata_file = set_path / "metadata.json"

    if not image_file.exists() or not metadata_file.exists():
        return

    # Create output directory
    output_dir = set_path / "sam_analysis"
    if output_dir.exists() and force:
        print(f"  Forcing re-analysis: Deleting {output_dir}")
        shutil.rmtree(output_dir)

    # Check if already processed
    if (output_dir / "metrics.json").exists() and (
        output_dir / "segments_summary.json"
    ).exists():
        print(f"  Skipping {set_path.name} (already processed)")
        return

    output_dir.mkdir(exist_ok=True)
    all_segments_dir = output_dir / "all_segments"
    all_segments_dir.mkdir(exist_ok=True)

    # Load metadata to get concepts
    with open(metadata_file, "r") as f:
        meta = json.load(f)

    concepts = meta.get("concepts", [])
    if not concepts:
        return

    # Load original image for masked results
    img_orig = Image.open(image_file).convert("RGB")
    img_np = np.array(img_orig)

    # 1. Run SAM
    print(f"  Running SAM on {image_file.name} (device={device})...")
    results = model.predict(image_file, conf=0.25, verbose=False, device=device)

    if not results or len(results) == 0:
        print(f"    No segments found in {set_path}")
        return

    # Extract segments
    masks = results[0].masks.data.cpu().numpy()  # [N, H, W]
    num_segments = masks.shape[0]

    # Save ALL segments
    print(f"    Saving {num_segments} segments...")
    for i in range(num_segments):
        s_mask = (masks[i] > 0.5).astype(np.uint8)

        # Save binary mask
        mask_path = all_segments_dir / f"segment_{i}.png"
        Image.fromarray(s_mask * 255).save(mask_path)

        # Save masked image
        masked_img_np = img_np.copy()
        masked_img_np[s_mask == 0] = 0
        Image.fromarray(masked_img_np).save(
            all_segments_dir / f"masked_segment_{i}.png"
        )

    metrics = {}

    # 2. Match each concept
    for concept in concepts:
        safe_concept = (
            concept.replace(" ", "_")
            .replace("'", "")
            .replace('"', "")
            .replace("/", "-")
        )
        concept_mask_path = set_path / f"mask_{safe_concept}.png"

        if not concept_mask_path.exists():
            continue

        print(f"    Matching concept: {concept}")

        c_mask_img = Image.open(concept_mask_path).convert("L")
        c_mask_np = np.array(c_mask_img) > 127

        # Load raw upscaled heatmap for sensitivity curve
        heatmap_upscaled_path = set_path / f"upscaled_heatmap_{safe_concept}.png"
        heatmap_np = None
        if heatmap_upscaled_path.exists():
            heatmap_np = (
                np.array(Image.open(heatmap_upscaled_path).convert("L")) / 255.0
            )

        # Greedy composition of segments to maximize IoU
        current_composite_mask = np.zeros_like(c_mask_np, dtype=bool)
        selected_indices = []
        best_iou = 0.0

        # Pre-calculate boolean masks for all segments to speed up loop
        segment_masks = [masks[i] > 0.5 for i in range(num_segments)]

        while True:
            improved = False
            best_temp_iou = best_iou
            best_temp_idx = -1

            for i in range(num_segments):
                if i in selected_indices:
                    continue

                # Check if adding this segment improves IoU
                temp_mask = np.logical_or(current_composite_mask, segment_masks[i])
                temp_iou = calculate_iou(c_mask_np, temp_mask)

                if temp_iou > best_temp_iou:
                    best_temp_iou = temp_iou
                    best_temp_idx = i

            if best_temp_idx != -1:
                current_composite_mask = np.logical_or(
                    current_composite_mask, segment_masks[best_temp_idx]
                )
                selected_indices.append(best_temp_idx)
                best_iou = best_temp_iou
                improved = True

            if not improved:
                break

        if selected_indices:
            # Save the composite matching mask
            best_mask_img = Image.fromarray(
                (current_composite_mask * 255).astype(np.uint8)
            )
            best_mask_img.save(output_dir / f"matched_mask_{safe_concept}.png")

            # Save the composite matching masked image
            matched_masked_np = img_np.copy()
            matched_masked_np[current_composite_mask == 0] = 0
            Image.fromarray(matched_masked_np).save(
                output_dir / f"matched_masked_image_{safe_concept}.png"
            )

            metrics[concept] = {
                "iou": best_iou,
                "coverage": calculate_coverage(c_mask_np, current_composite_mask),
                "precision": calculate_precision(c_mask_np, current_composite_mask),
                "recall": calculate_recall(c_mask_np, current_composite_mask),
                "segment_indices": [int(idx) for idx in selected_indices],
                "num_segments_used": len(selected_indices),
            }

            # Calculate IoU sensitivity curve if heatmap is available
            if heatmap_np is not None:
                iou_curve = {}
                thresholds = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
                for t in thresholds:
                    t_mask = heatmap_np > t
                    t_iou = calculate_iou(t_mask, current_composite_mask)
                    iou_curve[str(t)] = float(t_iou)
                metrics[concept]["iou_curve"] = iou_curve

    # 3. Calculate Concept Collisions (Overlap between concepts)
    collisions = {}
    concept_list = list(metrics.keys())
    for i in range(len(concept_list)):
        for j in range(i + 1, len(concept_list)):
            c1, c2 = concept_list[i], concept_list[j]

            def get_safe(c):
                return (
                    c.replace(" ", "_")
                    .replace("'", "")
                    .replace('"', "")
                    .replace("/", "-")
                )

            m1_path = output_dir / f"matched_mask_{get_safe(c1)}.png"
            m2_path = output_dir / f"matched_mask_{get_safe(c2)}.png"

            if m1_path.exists() and m2_path.exists():
                m1 = np.array(Image.open(m1_path).convert("L")) > 127
                m2 = np.array(Image.open(m2_path).convert("L")) > 127
                overlap_iou = calculate_iou(m1, m2)
                collisions[f"{c1}_vs_{c2}"] = float(overlap_iou)

    # Calculate overall SAM coverage
    all_masks_union = np.zeros(masks.shape[1:], dtype=bool)
    for i in range(num_segments):
        all_masks_union = np.logical_or(all_masks_union, masks[i] > 0.5)

    overall_coverage_percent = (np.sum(all_masks_union) / all_masks_union.size) * 100

    # Save coverage gap visualization
    coverage_vis = img_np.copy()
    coverage_vis[~all_masks_union] = [255, 0, 0]  # Red for gaps
    Image.fromarray(coverage_vis).save(output_dir / "debug_coverage_gaps.png")

    # Save metrics
    with open(output_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

    with open(output_dir / "segments_summary.json", "w") as f:
        json.dump(
            {
                "num_segments": int(num_segments),
                "image_size": list(masks.shape[1:]),
                "overall_coverage_percent": float(overall_coverage_percent),
                "concept_collisions": collisions,
            },
            f,
            indent=4,
        )


def main():
    parser = argparse.ArgumentParser(
        description="Run SAM analysis on generated images."
    )
    parser.add_argument(
        "--root",
        type=str,
        default="results/object_analysis",
        help="Root directory for results.",
    )
    parser.add_argument(
        "--model", type=str, default="sam2_b.pt", help="SAM model weights."
    )
    parser.add_argument(
        "--single-dir", type=str, help="Process only a specific directory."
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="Delete existing SAM analysis and regenerate everything.",
    )
    args = parser.parse_args()

    print(f"Loading SAM model: {args.model}")
    model = SAM(args.model)

    # Determine device
    import torch

    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "0"
    else:
        device = "cpu"
    print(f"Using device: {device}")

    root_path = Path(args.root)

    if args.single_dir:
        print(f"Processing single directory: {args.single_dir}")
        process_directory(Path(args.single_dir), model, device, force=args.force)
        return
    else:
        print(f"Scanning {root_path} for set_* directories...")
        set_dirs = []
        for root, dirs, files in os.walk(root_path):
            if Path(root).name.startswith("set_"):
                set_dirs.append(root)

        print(f"Found {len(set_dirs)} directories. Starting analysis...")
        for s_dir in tqdm(sorted(set_dirs)):
            process_directory(Path(s_dir), model, device, force=args.force)


if __name__ == "__main__":
    main()
