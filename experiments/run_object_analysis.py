"""
    Experiment script for Object Category Analysis using Concept Attention with Flux.
    
    This script generates images for selected challenging object categories and 
    saves the corresponding concept attention heatmaps.
"""

import os
import torch
import numpy as np
import json
from PIL import Image
from concept_attention import ConceptAttentionFluxPipeline
from tqdm import tqdm

def save_experiment_results(
    output_dir, 
    prompt, 
    seed, 
    pipeline_output, 
    concepts
):
    """
    Saves the image, heatmaps, and metadata for a single experiment run.
    """
    # Create directory for this specific run
    run_dir = os.path.join(output_dir, f"seed_{seed}")
    os.makedirs(run_dir, exist_ok=True)

    # Save original image
    pipeline_output.image.save(os.path.join(run_dir, "image.png"))

    # Save heatmaps
    for concept, heatmap in zip(concepts, pipeline_output.concept_heatmaps):
        heatmap.save(os.path.join(run_dir, f"heatmap_{concept}.png"))
        
    # Save metadata
    metadata = {
        "prompt": prompt,
        "concepts": concepts,
        "seed": seed,
        "width": 1024,
        "height": 1024
    }
    with open(os.path.join(run_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=4)

    # Optionally save raw attention maps (numpy) if needed for quantitative analysis later
    # We can save them as .npy files. 
    # Note: These can be large, so we might only want to do this if strictly necessary.
    # For now, let's leave it out or make it optional.

def main():
    # 1. Setup Device
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")

    # 2. Initialize Model
    # Note: Using float16 or bfloat16 might be better for memory on Mac, 
    # but the example used default (likely float32/bfloat16 mixed handled by library).
    pipeline = ConceptAttentionFluxPipeline(
        model_name="flux-schnell",
        device=device
    )
    
    # 3. Define Experiments
    experiments = [
        {
            "category": "hands_fingers",
            "prompt": "A close up photo of a person's hands holding a red apple",
            "concepts": ["hands", "fingers", "apple", "red"]
        },
        {
            "category": "text_typography",
            "prompt": "A neon sign that says 'CONCEPT'",
            "concepts": ["neon sign", "text", "CONCEPT", "light"]
        },
        {
            "category": "reflection_transparency",
            "prompt": "A glass of water on a wooden table with a reflection",
            "concepts": ["glass", "water", "reflection", "table", "wood"]
        },
        {
            "category": "countable_objects",
            "prompt": "A jar full of colorful marbles",
            "concepts": ["jar", "marbles", "colorful", "glass"]
        },
        {
            "category": "spatial_relations",
            "prompt": "A cat sleeping under a wooden chair",
            "concepts": ["cat", "chair", "under", "wood", "floor"]
        }
    ]

    # 4. Run Experiments
    base_output_dir = "results/object_analysis"
    # Use fewer seeds for initial testing to save time. 
    # Restore to [0, 1, 2, 3, 4] for full experiment.
    seeds = [0]
    
    print(f"Starting experiments. Results will be saved to {base_output_dir}")

    for exp in experiments:
        category = exp["category"]
        prompt = exp["prompt"]
        concepts = exp["concepts"]
        
        print(f"\nProcessing Category: {category}")
        print(f"Prompt: {prompt}")
        
        category_dir = os.path.join(base_output_dir, category)
        
        for seed in tqdm(seeds, desc=f"Seeds for {category}"):
            try:
                pipeline_output = pipeline.generate_image(
                    prompt=prompt,
                    concepts=concepts,
                    width=1024,
                    height=1024,
                    layer_indices=[16, 17, 18], # Specific layers often used for semantic info
                    num_inference_steps=10,      # Flux-Schnell default
                    timesteps=list(range(0, 4)),
                    seed=seed
                )
                
                save_experiment_results(
                    output_dir=category_dir,
                    prompt=prompt,
                    seed=seed,
                    pipeline_output=pipeline_output,
                    concepts=concepts
                )
                
            except Exception as e:
                print(f"Error processing {category} seed {seed}: {e}")

    print("\nAll experiments completed.")

if __name__ == "__main__":
    main()
