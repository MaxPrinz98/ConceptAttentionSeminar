"""
Enhanced Experiment script for Object and Attribute Analysis
using Concept Attention with Flux.

This script investigates:
1. Challenging object categories (hands, text, etc.)
2. Attribute concepts (color, texture, spatial relations)
3. Failure cases and hallucinations (compositional, anatomical, logical)

It supports running multiple concept sets for the same prompt to analyze
how attention behaves under different queries.

Experiment definitions are loaded from a JSON config file
(default: seminar_project/experiments.json).
"""

import os
import argparse
import torch
import json
from concept_attention import ConceptAttentionFluxPipeline
from tqdm import tqdm
from datetime import datetime
from pathlib import Path


def concepts_to_dir_name(concepts):
    """Convert a list of concepts into a directory name like 'set_apple_red_cat'."""
    safe_parts = []
    for c in concepts:
        safe = (
            c.lower()
            .replace(" ", "_")
            .replace("'", "")
            .replace('"', "")
            .replace("/", "-")
        )
        safe_parts.append(safe)
    return "set_" + "_".join(safe_parts)


def save_experiment_results(
    output_dir,
    group_name,
    case_name,
    prompt,
    seed,
    pipeline_output,
    concepts,
):
    """
    Saves the image, heatmaps, and metadata with a clean structure.
    Directory: {output_dir}/{group_name}/{case_name}/seed_{seed}/{set_dir_name}/
    """
    # Create directory structure
    set_dir_name = concepts_to_dir_name(concepts)
    run_dir = os.path.join(
        output_dir, group_name, case_name, f"seed_{seed}", set_dir_name
    )
    os.makedirs(run_dir, exist_ok=True)

    # Save original image
    pipeline_output.image.save(os.path.join(run_dir, "image.png"))

    # Save heatmaps
    for concept, heatmap in zip(concepts, pipeline_output.concept_heatmaps):
        safe_concept = (
            concept.replace(" ", "_")
            .replace("'", "")
            .replace('"', "")
            .replace("/", "-")
        )
        heatmap.save(os.path.join(run_dir, f"heatmap_{safe_concept}.png"))

    # Save metadata
    metadata = {
        "category": group_name,
        "case_name": case_name,
        "prompt": prompt,
        "concepts": concepts,
        "concept_set_name": set_dir_name,
        "seed": seed,
        "timestamp": datetime.now().isoformat(),
        "model": "flux-schnell",
        "parameters": {
            "width": 1024,
            "height": 1024,
            "layer_indices": [16, 17, 18],
            "num_inference_steps": 4,
            "timesteps": [0, 1, 2, 3],
        },
    }
    with open(os.path.join(run_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=4)


def load_experiments(config_path):
    """Load experiment definitions from a JSON file."""
    config_path = Path(config_path)
    if not config_path.exists():
        print(f"Error: Config file not found: {config_path}")
        raise SystemExit(1)

    with open(config_path, "r") as f:
        experiments = json.load(f)

    # Count totals for info
    total_cases = sum(len(g["cases"]) for g in experiments)
    total_sets = sum(len(c["concept_sets"]) for g in experiments for c in g["cases"])
    print(
        f"Loaded {len(experiments)} groups, {total_cases} cases, {total_sets} concept sets from {config_path}"
    )
    return experiments


def main():
    parser = argparse.ArgumentParser(
        description="Generate images and concept-attention heatmaps using Flux.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="seminar_project/experiments.json",
        help="Path to the experiments JSON config file (default: seminar_project/experiments.json)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/object_analysis",
        help="Output directory for results (default: results/object_analysis)",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[0],
        help="Seeds to run (default: 0). Example: --seeds 0 1 2",
    )
    args = parser.parse_args()

    # 1. Setup Device
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"Using device: {device}")

    # 2. Load experiment config
    experiments = load_experiments(args.config)

    # 3. Initialize Model
    pipeline = ConceptAttentionFluxPipeline(model_name="flux-schnell", device=device)

    # 4. Run Experiments
    base_output_dir = args.output
    seeds = args.seeds

    print(f"Starting experiments. Results will be saved to {base_output_dir}")

    for group in experiments:
        group_name = group["group"]

        for case in group["cases"]:
            case_name = case["name"]
            prompt = case["prompt"]
            concept_sets = case["concept_sets"]

            print(f"\nGroup: {group_name} | Case: {case_name}")
            print(f"Prompt: {prompt}")

            for concepts in concept_sets:
                print(f"  Concept Set: {concepts}")

                for seed in tqdm(seeds, desc="    Processing seeds"):
                    try:
                        # Check if already computed
                        set_dir_name = concepts_to_dir_name(concepts)
                        run_dir = os.path.join(
                            base_output_dir,
                            group_name,
                            case_name,
                            f"seed_{seed}",
                            set_dir_name,
                        )
                        if os.path.exists(run_dir):
                            print(
                                f"    Skipping {case_name} seed {seed} {set_dir_name} (already exists)"
                            )
                            continue

                        # Flux generation with concept attention tracking
                        pipeline_output = pipeline.generate_image(
                            prompt=prompt,
                            concepts=concepts,
                            width=1024,
                            height=1024,
                            layer_indices=[16, 17, 18],
                            num_inference_steps=4,
                            timesteps=list(range(0, 4)),
                            seed=seed,
                        )

                        save_experiment_results(
                            output_dir=base_output_dir,
                            group_name=group_name,
                            case_name=case_name,
                            prompt=prompt,
                            seed=seed,
                            pipeline_output=pipeline_output,
                            concepts=concepts,
                        )

                    except Exception as e:
                        print(f"    Error processing {case_name} seed {seed}: {e}")

    print("\nAll experiments completed.")


if __name__ == "__main__":
    main()
