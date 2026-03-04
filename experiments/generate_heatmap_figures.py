import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from pathlib import Path
import math


def create_publication_figure(results_dir, output_dir, max_concepts=6):
    """
    Creates publication-ready grids of heatmaps (like Figure 3 in paper).
    For each prompt, it shows the prompt text, an arrow, the original image,
    and then the heatmaps, tightly packed with the concept text overlay.
    """
    results_dir = Path(results_dir)
    output_dir = Path(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    print(f"Scanning {results_dir} for experiments...")

    # Group by case
    cases = {}
    for root, dirs, files in os.walk(results_dir):
        if Path(root).name.startswith("set_"):
            set_path = Path(root)
            seed_dir = set_path.parent
            case_dir = seed_dir.parent
            group_dir = case_dir.parent

            case_key = f"{group_dir.name}_{case_dir.name}_{seed_dir.name}"

            if case_key not in cases:
                cases[case_key] = {
                    "case_name": case_dir.name,
                    "group_name": group_dir.name,
                    "seed": seed_dir.name,
                    "sets": [],
                }
            cases[case_key]["sets"].append(set_path)

    # Process each case
    for case_key, case_info in cases.items():
        case_name = case_info["case_name"]

        # Sort sets numerically
        sets = sorted(case_info["sets"], key=lambda x: int(x.name.split("_")[1]))

        # Create a figure for the entire case (all sets combined as rows)
        set_data_list = []
        max_concepts_in_case = 0

        for set_path in sets:
            metadata_file = set_path / "metadata.json"
            if not metadata_file.exists():
                continue

            with open(metadata_file, "r") as f:
                meta = json.load(f)

            prompt = meta.get("prompt", "")
            concepts = meta.get("concepts", [])
            concepts = concepts[:max_concepts]
            max_concepts_in_case = max(max_concepts_in_case, len(concepts))

            img_path = set_path / "image.png"
            if not img_path.exists():
                continue

            heatmaps = []
            for concept in concepts:
                safe_concept = (
                    concept.replace(" ", "_")
                    .replace("'", "")
                    .replace('"', "")
                    .replace("/", "-")
                )
                # Try upscaled first, then original
                heatmap_path = set_path / f"upscaled_heatmap_{safe_concept}.png"
                if not heatmap_path.exists():
                    heatmap_path = set_path / f"heatmap_{safe_concept}.png"

                if heatmap_path.exists():
                    heatmaps.append((concept, heatmap_path))

            if heatmaps:
                set_data_list.append(
                    {"prompt": prompt, "image_path": img_path, "heatmaps": heatmaps}
                )

        if not set_data_list:
            continue

        print(
            f"Generating LaTeX figure for {case_name} with {len(set_data_list)} rows..."
        )

        rows = len(set_data_list)
        # Columns: 1 Text, 1 Image, N Heatmaps
        cols = 1 + 1 + max_concepts_in_case

        fig_height = rows * 1.8
        fig_width = cols * 1.8

        fig = plt.figure(figsize=(fig_width, fig_height), dpi=300)

        width_ratios = [1.5, 1.2] + [1.0] * max_concepts_in_case
        gs = fig.add_gridspec(
            rows, len(width_ratios), width_ratios=width_ratios, wspace=0.05, hspace=0.1
        )

        # --- Draw Prompt Text (Spanning all rows) ---
        ax_text = fig.add_subplot(gs[:, 0])
        ax_text.axis("off")

        first_prompt = set_data_list[0]["prompt"]
        import textwrap

        wrapped_prompt = "\n".join(textwrap.wrap(first_prompt, width=20))

        ax_text.text(
            0.5,
            0.5,
            wrapped_prompt,
            ha="center",
            va="center",
            fontsize=12,
            color="black",
            fontfamily="sans-serif",
        )

        for r, data in enumerate(set_data_list):
            # --- Draw Original Image (Only first row, spanning if needed, or just top) ---
            if r == 0:
                # We can make the image span all rows too, or just sit in the top row.
                # Figure 3 usually shows the image once. We'll span it to center it, or put it in the middle.
                ax_img = fig.add_subplot(gs[:, 1])
                ax_img.axis("off")
                try:
                    img = mpimg.imread(data["image_path"])
                    ax_img.imshow(img)
                except Exception as e:
                    print(f"Error loading {data['image_path']}: {e}")

            # --- Draw Heatmaps ---
            for c in range(max_concepts_in_case):
                ax_hm = fig.add_subplot(gs[r, 2 + c])
                ax_hm.axis("off")

                if c < len(data["heatmaps"]):
                    concept, hm_path = data["heatmaps"][c]
                    try:
                        hm = mpimg.imread(hm_path)
                        ax_hm.imshow(hm)
                        # Add text overlay
                        ax_hm.text(
                            0.5,
                            0.95,
                            concept,
                            ha="center",
                            va="top",
                            transform=ax_hm.transAxes,
                            color="white",
                            fontsize=12,
                            fontfamily="sans-serif",
                            bbox=dict(
                                facecolor="black", alpha=0.3, edgecolor="none", pad=1
                            ),
                        )
                    except Exception as e:
                        print(f"Error loading {hm_path}: {e}")

        # Global Title (optional)
        # fig.suptitle(f"Case: {case_name}", y=0.98, fontsize=12, fontweight='bold')

        # Adjust layout to remove excessive padding
        plt.subplots_adjust(left=0.02, right=0.98, top=0.95, bottom=0.05)

        # Save as high-res PNG and PDF
        out_png = output_dir / f"{case_key}.png"
        out_pdf = output_dir / f"{case_key}.pdf"

        plt.savefig(out_png, bbox_inches="tight", transparent=False, pad_inches=0.05)
        plt.savefig(out_pdf, bbox_inches="tight", transparent=False, pad_inches=0.05)
        plt.close(fig)
        print(f"Saved {out_png} and {out_pdf}")


if __name__ == "__main__":
    RESULTS_DIR = "results/object_analysis"
    OUTPUT_DIR = "experiments/results_heatmap_grids"

    # Optional: ensure matplotlib uses a good backend
    import matplotlib

    matplotlib.use("Agg")

    create_publication_figure(RESULTS_DIR, OUTPUT_DIR, max_concepts=6)
    print("Done generating publication figures.")
