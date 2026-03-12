"""
Pipeline orchestrator for the Concept Attention evaluation workflow.

Runs all processing steps in order.  Use ``--help`` to see available
flags, or ``--list-steps`` to see what each step does.

Usage examples
--------------
    # Run everything
    uv run seminar_project/pipeline.py

    # Run only the reporting steps (4-6)
    uv run seminar_project/pipeline.py --from-step 4

    # Rerun SAM + everything after it
    uv run seminar_project/pipeline.py --from-step 3 --force

    # Run only step 6 (HTML generation)
    uv run seminar_project/pipeline.py --only-step 6

    # Skip the heavy GPU generation, run everything else
    uv run seminar_project/pipeline.py --skip-steps 1
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

# ── Step definitions ──────────────────────────────────────────────────
STEPS = [
    {
        "num": 1,
        "name": "Generate Images & Heatmaps",
        "description": (
            "Runs the Flux pipeline to generate 1024×1024 images and extract "
            "concept-attention heatmaps for every experiment defined in the "
            "experiment list.  (GPU-heavy, uses Flux-schnell)"
        ),
        "script": "seminar_project/run_object_analysis.py",
        "extra_args": [],
    },
    {
        "num": 2,
        "name": "Upscale Heatmaps & Create Masks",
        "description": (
            "Upscales the raw heatmaps to 1024×1024, applies a threshold to "
            "create binary masks, and produces masked images.  (CPU-only, fast)"
        ),
        "script": "seminar_project/process_heatmaps.py",
        "extra_args": [],
    },
    {
        "num": 3,
        "name": "SAM Segmentation & Metrics",
        "description": (
            "Runs SAM auto-segmentation on every generated image, greedily "
            "matches SAM segments to concept masks, and computes IoU / "
            "Precision / Recall metrics.  (GPU, uses SAM2)"
        ),
        "script": "seminar_project/run_sam_analysis.py",
        "extra_args": [],
    },
    {
        "num": 4,
        "name": "Generate Grid Figures",
        "description": (
            "Creates publication-ready matplotlib grids showing the prompt, "
            "generated image, and per-concept heatmaps side by side.  "
            "Output: results/results_heatmap_grids/  (CPU-only)"
        ),
        "script": "seminar_project/generate_heatmap_figures.py",
        "extra_args": [],
    },
    {
        "num": 5,
        "name": "Extract Embeddings & UMAP",
        "description": (
            "Loads the T5 encoder from Flux, extracts embeddings for every "
            "unique concept, and runs UMAP dimensionality reduction.  "
            "Output: results/object_analysis/concept_umap.json  (GPU)"
        ),
        "script": "seminar_project/extract_embeddings.py",
        "extra_args": [],
    },
    {
        "num": 6,
        "name": "Generate HTML Report",
        "description": (
            "Assembles the final interactive evaluation dashboard from all "
            "previously generated data (metrics, grids, UMAP, images).  "
            "Output: results/object_analysis/evaluation_gallery.html  (CPU-only, fast)"
        ),
        "script": "seminar_project/evaluate_object_analysis_html.py",
        "extra_args": [],
    },
]


def print_steps():
    """Pretty-print all pipeline steps."""
    print("\n📋  Pipeline Steps\n" + "─" * 60)
    for step in STEPS:
        print(f"\n  Step {step['num']}:  {step['name']}")
        print(f"          {step['description']}")
        print(f"          Script: {step['script']}")
    print()


def run_step(step, force=False):
    """Execute a single pipeline step as a subprocess."""
    script = step["script"]
    args = ["uv", "run", script] + step["extra_args"]
    if force and step["num"] in [2, 3]:
        args.append("--force")

    header = f"Step {step['num']}: {step['name']}"
    print(f"\n{'═' * 60}")
    print(f"▶  {header}")
    print(f"   {step['script']}")
    print(f"{'═' * 60}\n")

    t0 = time.time()
    result = subprocess.run(args, cwd=str(Path(__file__).resolve().parent.parent))
    elapsed = time.time() - t0

    if result.returncode != 0:
        print(f"\n{header}  FAILED  (exit code {result.returncode})")
        sys.exit(result.returncode)

    mins, secs = divmod(int(elapsed), 60)
    print(f"\n{header}  completed in {mins}m {secs}s")
    return elapsed


def main():
    parser = argparse.ArgumentParser(
        description="Concept Attention evaluation pipeline — runs all 6 processing steps in order.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  uv run seminar_project/pipeline.py                  # run everything\n"
            "  uv run seminar_project/pipeline.py --from-step 4    # reporting only\n"
            "  uv run seminar_project/pipeline.py --only-step 6    # HTML only\n"
            "  uv run seminar_project/pipeline.py --skip-steps 1 5 # skip generation & UMAP\n"
            "  uv run seminar_project/pipeline.py --list-steps     # show step descriptions\n"
        ),
    )
    parser.add_argument(
        "--list-steps",
        action="store_true",
        help="Print a description of each step and exit.",
    )
    parser.add_argument(
        "--from-step",
        type=int,
        metavar="N",
        default=1,
        help="Start from step N (1-6), skipping earlier steps.  (default: 1)",
    )
    parser.add_argument(
        "--to-step",
        type=int,
        metavar="N",
        default=6,
        help="Stop after step N (1-6).  (default: 6)",
    )
    parser.add_argument(
        "--only-step",
        type=int,
        metavar="N",
        help="Run only step N.",
    )
    parser.add_argument(
        "--skip-steps",
        type=int,
        nargs="+",
        metavar="N",
        default=[],
        help="Skip specific step numbers, e.g. --skip-steps 1 5",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force regeneration (passed to SAM analysis as --force).",
    )
    args = parser.parse_args()

    if args.list_steps:
        print_steps()
        return

    # Determine which steps to run
    if args.only_step:
        steps_to_run = [s for s in STEPS if s["num"] == args.only_step]
    else:
        steps_to_run = [
            s
            for s in STEPS
            if args.from_step <= s["num"] <= args.to_step
            and s["num"] not in args.skip_steps
        ]

    if not steps_to_run:
        print("No steps selected. Use --list-steps to see available steps.")
        return

    step_nums = ", ".join(str(s["num"]) for s in steps_to_run)
    print(f"\n Running pipeline steps: [{step_nums}]")
    if args.force:
        print("   --force is set")

    total_t0 = time.time()
    for step in steps_to_run:
        run_step(step, force=args.force)

    total_elapsed = time.time() - total_t0
    mins, secs = divmod(int(total_elapsed), 60)
    print(f"\n{'═' * 60}")
    print(f"🏁  Pipeline complete!  Total time: {mins}m {secs}s")
    print(f"{'═' * 60}\n")


if __name__ == "__main__":
    main()
