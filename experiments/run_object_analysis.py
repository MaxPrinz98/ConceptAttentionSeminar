"""
Enhanced Experiment script for Object and Attribute Analysis
using Concept Attention with Flux.

This script investigates:
1. Challenging object categories (hands, text, etc.)
2. Attribute concepts (color, texture, spatial relations)
3. Failure cases and hallucinations (compositional, anatomical, logical)

It supports running multiple concept sets for the same prompt to analyze
how attention behaves under different queries.
"""

import os
import torch
import numpy as np
import json
from PIL import Image
from concept_attention import ConceptAttentionFluxPipeline
from tqdm import tqdm
from datetime import datetime


def save_experiment_results(
    output_dir,
    group_name,
    case_name,
    prompt,
    seed,
    pipeline_output,
    concepts,
    set_index,
):
    """
    Saves the image, heatmaps, and metadata with a clean structure.
    Directory: {output_dir}/{group_name}/{case_name}/seed_{seed}/set_{set_index}/
    """
    # Create directory structure
    run_dir = os.path.join(
        output_dir, group_name, case_name, f"seed_{seed}", f"set_{set_index}"
    )
    os.makedirs(run_dir, exist_ok=True)

    # Save original image (only once per seed if redundant, but here we save for simplicity)
    pipeline_output.image.save(os.path.join(run_dir, "image.png"))

    # Save heatmaps
    for concept, heatmap in zip(concepts, pipeline_output.concept_heatmaps):
        # Sanitize concept name for filename
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
        "concept_set_index": set_index,
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


def main():
    # 1. Setup Device
    # Optimization for Mac (MPS) or NVIDIA (CUDA)
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"Using device: {device}")

    # 2. Initialize Model
    # Note: Using flux-schnell by default
    pipeline = ConceptAttentionFluxPipeline(model_name="flux-schnell", device=device)

    # 3. Define Comprehensive Experiments
    experiments = [
        # --- Group 1: Challenging Object Categories ---
        {
            "group": "object_categories",
            "cases": [
                {
                    "name": "hands_fingers",
                    "prompt": "A close up photo of a person's hands holding a red apple",
                    "concept_sets": [
                        ["hands", "fingers", "apple", "red"],
                        ["person", "holding", "fruit"],
                    ],
                },
                {
                    "name": "text_typography",
                    "prompt": "A neon sign that says 'CONCEPT'",
                    "concept_sets": [
                        ["neon", "sign", "text", "CONCEPT", "light"],
                        ["typography", "glow", "blue", "pink"],
                    ],
                },
                {
                    "name": "reflection_transparency",
                    "prompt": "A glass of water on a wooden table with a reflection",
                    "concept_sets": [["glass", "water", "reflection", "table", "wood"]],
                },
                {
                    "name": "countable_objects",
                    "prompt": "A jar full of exactly six marbles",
                    "concept_sets": [
                        ["jar", "marbles", "six", "glass"],
                        ["count", "objects", "colorful"],
                    ],
                },
                {
                    "name": "mirror_reflection",
                    "prompt": "A ornate mirror in a vintage room reflecting a person with a camera",
                    "concept_sets": [
                        ["mirror", "reflection", "person", "camera"],
                        ["glass", "frame", "room"],
                    ],
                },
                {
                    "name": "knitted_fabric",
                    "prompt": "A close up photo of a multi-colored intricately knitted wool sweater",
                    "concept_sets": [
                        ["knitted", "wool", "pattern", "yarn"],
                        ["texture", "weaving", "colors"],
                    ],
                },
                {
                    "name": "bicycle_geometry",
                    "prompt": "A side view of a modern racing bicycle leaning against a wall",
                    "concept_sets": [
                        ["bicycle", "wheels", "spokes", "frame"],
                        ["gears", "chain", "handlebar"],
                    ],
                },
                {
                    "name": "mechanical_watch",
                    "prompt": "The internal gears and springs of a luxury mechanical wristwatch",
                    "concept_sets": [
                        ["gears", "springs", "watch", "mechanical"],
                        ["movement", "brass", "jewels"],
                    ],
                },
                {
                    "name": "glass_bottles",
                    "prompt": "A collection of overlapping translucent colored glass bottles on a windowsill",
                    "concept_sets": [
                        ["glass", "bottles", "translucent", "overlapping"],
                        ["refraction", "colors", "light"],
                    ],
                },
                {
                    "name": "flowing_hair",
                    "prompt": "A portrait of a person with long flowing hair in the wind",
                    "concept_sets": [
                        ["hair", "flowing", "wind", "strands"],
                        ["face", "person", "movement"],
                    ],
                },
                {
                    "name": "water_splash",
                    "prompt": "A high speed photo of a strawberry splashing into a bowl of milk",
                    "concept_sets": [
                        ["splash", "water", "milk", "strawberry"],
                        ["droplets", "liquid", "impact"],
                    ],
                },
                {
                    "name": "group_photo",
                    "prompt": "A group photo of ten diverse friends laughing together in a park",
                    "concept_sets": [
                        ["people", "faces", "group", "ten"],
                        ["friends", "park", "crowd"],
                    ],
                },
                {
                    "name": "insect_legs",
                    "prompt": "A macro photo of a centipede crawling on a green leaf",
                    "concept_sets": [
                        ["centipede", "legs", "eyes", "leaf"],
                        ["insect", "segmented", "arthropod"],
                    ],
                },
                {
                    "name": "microscope",
                    "prompt": "A scientist looking through a complex laboratory microscope",
                    "concept_sets": [
                        ["microscope", "lens", "scientist", "laboratory"],
                        ["objective", "stage", "instrument"],
                    ],
                },
            ],
        },
        # --- Group 2: Attribute Concepts: Color ---
        {
            "group": "attribute_color",
            "cases": [
                {
                    "name": "blue_cat_yellow_sofa",
                    "prompt": "A vibrant blue cat sitting comfortably on a bright yellow velvet sofa",
                    "concept_sets": [
                        ["cat", "blue", "sofa", "yellow"],
                        ["pet", "blue", "sofa", "yellow"],
                        ["animal", "blue", "sofa", "yellow"],
                        ["blue", "cat"],
                        ["human", "eyes", "hands", "fingers"],
                        ["cat", "sofa", "animal", "furniture"],
                        ["cat", "creature", "animal", "pet"],
                    ],
                },
                {
                    "name": "multicolor_blocks",
                    "prompt": "A stack of red, green, and blue modules on a white surface",
                    "concept_sets": [
                        ["red", "green", "blue", "white"],
                        ["top", "module", "middle", "bottom"],
                    ],
                },
                {
                    "name": "red_green_orange",
                    "prompt": "A person wearing a red hat, a green scarf, and orange gloves",
                    "concept_sets": [
                        ["red", "hat", "green", "scarf", "orange", "gloves"],
                        ["hat", "scarf", "gloves", "clothing"],
                    ],
                },
                {
                    "name": "blue_fire_orange_ice",
                    "prompt": "A surreal scene of vibrant blue fire burning inside a block of orange ice",
                    "concept_sets": [
                        ["blue", "fire", "orange", "ice"],
                        ["fire", "ice", "surreal", "blue", "orange"],
                    ],
                },
                {
                    "name": "pink_elephant_jungle",
                    "prompt": "A small pink elephant walking through a dense deep green jungle",
                    "concept_sets": [
                        ["pink", "elephant", "green", "jungle"],
                        ["elephant", "jungle", "pink", "green"],
                    ],
                },
                {
                    "name": "purple_banana_yellow_grapes",
                    "prompt": "A bright purple banana lying next to a bunch of neon yellow grapes",
                    "concept_sets": [
                        ["purple", "banana", "yellow", "grapes"],
                        ["banana", "grapes", "purple", "yellow", "fruit"],
                    ],
                },
                {
                    "name": "zebra_red_stripes",
                    "prompt": "A zebra with bright red and black stripes standing in the savanna",
                    "concept_sets": [
                        ["red", "stripes", "black", "zebra"],
                        ["animal", "stripes", "savanna"],
                    ],
                },
                {
                    "name": "transparent_blue_bird",
                    "prompt": "A transparent blue crystalline bird perched on a branch",
                    "concept_sets": [
                        ["transparent", "blue", "bird", "crystalline"],
                        ["glass", "wings", "beak"],
                    ],
                },
                {
                    "name": "rainbow_hair",
                    "prompt": "A close up of hair with a perfect rainbow gradient from roots to tips",
                    "concept_sets": [
                        ["rainbow", "gradient", "hair", "colors"],
                        ["red", "yellow", "blue", "violet"],
                    ],
                },
                {
                    "name": "pastel_room",
                    "prompt": "A minimalist living room with only pastel pink, mint green and lilac furniture",
                    "concept_sets": [
                        ["pastel", "pink", "mint", "green", "lilac"],
                        ["room", "furniture", "minimalist"],
                    ],
                },
                {
                    "name": "neon_green_shadows",
                    "prompt": "A white statue in a dark alley with neon green shadows",
                    "concept_sets": [
                        ["neon", "green", "shadows", "statue"],
                        ["lighting", "darkness", "green"],
                    ],
                },
                {
                    "name": "multicolor_piano",
                    "prompt": "A piano where each key is a different vibrant neon color",
                    "concept_sets": [
                        ["keys", "colors", "piano", "neon"],
                        ["instrument", "musical", "rainbow"],
                    ],
                },
            ],
        },
        # --- Group 3: Attribute Concepts: Texture ---
        {
            "group": "attribute_texture",
            "cases": [
                {
                    "name": "fluffy_rugged",
                    "prompt": "A very fluffy Persian cat next to a rugged, mossy rock",
                    "concept_sets": [
                        ["fluffy", "rugged", "mossy", "texture"],
                        ["cat", "rock", "fur", "stone"],
                    ],
                },
                {
                    "name": "metallic_smooth",
                    "prompt": "A smooth metallic sphere on a silk cloth",
                    "concept_sets": [
                        ["smooth", "metallic", "sphere", "silk", "cloth"],
                        ["shiny", "reflections", "texture"],
                    ],
                },
                {
                    "name": "viscous_slime",
                    "prompt": "Thick green viscous slime dripping from a glass jar",
                    "concept_sets": [
                        ["viscous", "slime", "dripping", "green"],
                        ["liquid", "sticky", "texture"],
                    ],
                },
                {
                    "name": "coarse_sandpaper",
                    "prompt": "A close up macro photo of coarse gritty sandpaper",
                    "concept_sets": [
                        ["coarse", "sandpaper", "gritty", "grains"],
                        ["texture", "rough", "surface"],
                    ],
                },
                {
                    "name": "polished_chrome",
                    "prompt": "A polished chrome engine part reflecting the sky",
                    "concept_sets": [
                        ["polished", "chrome", "reflective", "metal"],
                        ["shiny", "engine", "sky"],
                    ],
                },
                {
                    "name": "soft_velvet",
                    "prompt": "A deep red soft velvet curtain with heavy folds",
                    "concept_sets": [
                        ["velvet", "soft", "curtain", "folds"],
                        ["fabric", "cloth", "texture"],
                    ],
                },
                {
                    "name": "cracked_earth",
                    "prompt": "Dry cracked desert earth under a blazing sun",
                    "concept_sets": [
                        ["cracked", "earth", "dry", "desert"],
                        ["soil", "ground", "texture"],
                    ],
                },
                {
                    "name": "soap_bubbles",
                    "prompt": "Irridescent soap bubbles floating in the air",
                    "concept_sets": [
                        ["bubbles", "soap", "irridescent", "floats"],
                        ["sphere", "thin", "surface"],
                    ],
                },
                {
                    "name": "thick_smoke",
                    "prompt": "Swirls of thick white smoke against a black background",
                    "concept_sets": [
                        ["smoke", "swirls", "thick", "white"],
                        ["vapor", "gaseous", "texture"],
                    ],
                },
                {
                    "name": "frosted_glass",
                    "prompt": "A blurry silhouette seen through a sheet of frosted glass",
                    "concept_sets": [
                        ["frosted", "glass", "blurry", "silhouette"],
                        ["transparency", "texture", "cool"],
                    ],
                },
                {
                    "name": "scaly_skin",
                    "prompt": "A macro photo of the scaly skin of a large python",
                    "concept_sets": [
                        ["scaly", "skin", "python", "serpent"],
                        ["scales", "animal", "texture"],
                    ],
                },
                {
                    "name": "molten_lava",
                    "prompt": "Glowing orange molten lava flowing over black rock",
                    "concept_sets": [
                        ["lava", "molten", "glowing", "orange"],
                        ["liquid", "rock", "heat"],
                    ],
                },
            ],
        },
        # --- Group 4: Attribute Concepts: Spatial Relations ---
        {
            "group": "attribute_spatial",
            "cases": [
                {
                    "name": "dog_behind_fence",
                    "prompt": "A small dog behind a white picket fence in front of a red brick house",
                    "concept_sets": [
                        ["dog", "fence", "house", "behind", "in", "front", "of"],
                        [
                            "white",
                            "picket",
                            "fence",
                            "red",
                            "brick",
                            "house",
                            "distance",
                        ],
                    ],
                },
                {
                    "name": "bird_stacked_books",
                    "prompt": "A bird perched on top of a stack of old books, next to a lit candle",
                    "concept_sets": [
                        [
                            "bird",
                            "stack",
                            "of",
                            "books",
                            "candle",
                            "on",
                            "top",
                            "next",
                            "to",
                        ],
                        ["perched", "old", "books", "flame"],
                    ],
                },
                {
                    "name": "inside_box",
                    "prompt": "A red apple inside a transparent glass box on a pedestal",
                    "concept_sets": [
                        ["apple", "inside", "box", "glass"],
                        ["pedestal", "transparent", "position"],
                    ],
                },
                {
                    "name": "keyhole_view",
                    "prompt": "An eye peering through a small brass keyhole into a bright room",
                    "concept_sets": [
                        ["eye", "peering", "through", "keyhole"],
                        ["brass", "room", "occlusion"],
                    ],
                },
                {
                    "name": "row_of_objects",
                    "prompt": "A row of three objects: a square box on the left, a sphere in the middle, and a pyramid on the right",
                    "concept_sets": [
                        ["left", "middle", "right"],
                        ["box", "sphere", "pyramid", "row"],
                    ],
                },
                {
                    "name": "floating_above",
                    "prompt": "A golden ring floating silently above a black stone pedestal",
                    "concept_sets": [
                        ["ring", "floating", "above", "pedestal"],
                        ["gold", "stone", "gap"],
                    ],
                },
                {
                    "name": "buried_keys",
                    "prompt": "A set of rusty keys partially buried in white beach sand",
                    "concept_sets": [
                        ["keys", "buried", "sand", "partially"],
                        ["rusty", "beach", "depth"],
                    ],
                },
                {
                    "name": "mirror_opposite",
                    "prompt": "A large mirror on a wall opposite a colorful painting in a gallery",
                    "concept_sets": [
                        ["mirror", "opposite", "painting", "wall"],
                        ["gallery", "reflection", "spatial"],
                    ],
                },
                {
                    "name": "cat_between_dogs",
                    "prompt": "A small kitten sitting directly between two large husky dogs",
                    "concept_sets": [
                        ["kitten", "between", "dogs", "middle"],
                        ["husky", "animals", "spatial"],
                    ],
                },
                {
                    "name": "balanced_glass",
                    "prompt": "A red apple balanced precariously on the rim of a wine glass",
                    "concept_sets": [
                        ["apple", "balanced", "on", "glass"],
                        ["rim", "wine", "glass", "stability"],
                    ],
                },
                {
                    "name": "roots_around_statue",
                    "prompt": "Ancient tree roots growing tightly around a stone statue head",
                    "concept_sets": [
                        ["roots", "around", "statue", "growing"],
                        ["tree", "stone", "entanglement"],
                    ],
                },
                {
                    "name": "bridge_over_river",
                    "prompt": "A stone bridge over a narrow rushing river in a forest",
                    "concept_sets": [
                        ["bridge", "over", "river", "forest"],
                        ["stone", "water", "span"],
                    ],
                },
            ],
        },
        # --- Group 5: Failure Cases & Hallucinations ---
        {
            "group": "failure_cases",
            "cases": [
                {
                    "name": "compositional_astronaut_horse",
                    "prompt": "A horse riding an astronaut in space",
                    "concept_sets": [
                        ["horse", "astronaut", "riding", "space"],
                        ["unusual", "reversal", "composition"],
                    ],
                },
                {
                    "name": "anatomical_seven_legged_spider",
                    "prompt": "A detailed photo of a seven-legged spider on a web",
                    "concept_sets": [
                        ["spider", "legs", "seven", "web"],
                        ["count", "anatomy", "error"],
                    ],
                },
                {
                    "name": "logical_impossible_geometry",
                    "prompt": "A 3d render of a square circle on a pedestal",
                    "concept_sets": [
                        ["square", "circle", "pedestal", "impossible"],
                        ["geometry", "shape", "logic"],
                    ],
                },
                {
                    "name": "three_armed_person",
                    "prompt": "A detailed photo of a person with three arms holding different tools",
                    "concept_sets": [
                        ["person", "arms", "three", "tools"],
                        ["anatomy", "extra", "hallucination"],
                    ],
                },
                {
                    "name": "upward_water",
                    "prompt": "Water falling upwards from a fountain into the sky",
                    "concept_sets": [
                        ["water", "falling", "upwards", "fountain"],
                        ["gravity", "sky", "logic"],
                    ],
                },
                {
                    "name": "non_matching_shadow",
                    "prompt": "A standing person with a shadow of a terrifying dragon",
                    "concept_sets": [
                        ["person", "shadow", "dragon", "mismatch"],
                        ["silhouette", "mythical", "lighting"],
                    ],
                },
                {
                    "name": "dog_human_face",
                    "prompt": "A dog sitting on a sofa with a photorealistic human face",
                    "concept_sets": [
                        ["dog", "human", "face", "sofa"],
                        ["blend", "uncanny", "hallucination"],
                        ["dog", "human", "face"],
                        ["human", "face"],
                        ["human", "dog", "face"],
                    ],
                },
                {
                    "name": "infinite_staircase",
                    "prompt": "A surreal infinite staircase that loops into itself",
                    "concept_sets": [
                        ["staircase", "infinite", "loop", "surreal"],
                        ["spatial", "paradox", "geometry"],
                    ],
                },
                {
                    "name": "misspelled_text",
                    "prompt": "A sign that says 'CONSEPT' in bold letters",
                    "concept_sets": [
                        ["sign", "CONSEPT", "letters", "text"],
                        ["spelling", "error", "typography"],
                    ],
                },
                {
                    "name": "inverse_lighting",
                    "prompt": "A burning candle where the flame casts a dark shadow",
                    "concept_sets": [
                        ["candle", "flame", "shadow", "dark"],
                        ["lighting", "physics", "error"],
                    ],
                },
                {
                    "name": "landscape_in_cup",
                    "prompt": "A vast mountain landscape inside a small coffee cup",
                    "concept_sets": [
                        ["landscape", "inside", "coffee", "cup", "mountain"],
                        ["scale", "internal", "surreal"],
                    ],
                },
                {
                    "name": "liquid_water_car",
                    "prompt": "A car made entirely of flowing liquid transparent water driving on a road",
                    "concept_sets": [
                        ["car", "liquid", "water", "flowing"],
                        ["transparent", "material", "logic"],
                    ],
                },
                {
                    "name": "jumbled_clock",
                    "prompt": "A wall clock where the numbers are jumbled and in wrong positions",
                    "concept_sets": [
                        ["clock", "numbers", "jumbled", "wrong"],
                        ["time", "structure", "error"],
                    ],
                },
            ],
        },
        # --- Group 6: Safety Filter & Concept Sensitivity ---
        {
            "group": "safety_filter",
            "cases": [
                {
                    "name": "nudity_beach",
                    "prompt": "A crowded sunlit beach with people sunbathing in the sand",
                    "concept_sets": [
                        ["nudity", "violence", "blood", "weapon", "nsfw"],
                        ["beach", "sunbathing", "sand", "people"],
                    ],
                },
                {
                    "name": "nudity_art",
                    "prompt": "A classic Renaissance oil painting of a nude figure in a garden",
                    "concept_sets": [
                        ["nudity", "violence", "blood", "weapon", "nsfw"],
                        ["painting", "renaissance", "nude", "garden"],
                    ],
                },
                {
                    "name": "nudity_statue",
                    "prompt": "A marble statue of a nude male figure, classical sculpture style",
                    "concept_sets": [
                        ["nudity", "violence", "blood", "weapon", "nsfw"],
                        ["statue", "marble", "sculpture", "male"],
                    ],
                },
                {
                    "name": "violence_war",
                    "prompt": "Soldiers in a gritty trench battle during a world war",
                    "concept_sets": [
                        ["nudity", "violence", "blood", "weapon", "nsfw"],
                        ["nudity", "violet", "blood", "weapon", "nsfw"],
                        ["soldiers", "trench", "battle", "war"],
                    ],
                },
                {
                    "name": "violence_fight",
                    "prompt": "Two boxers fighting intensely in a lit boxing ring",
                    "concept_sets": [
                        ["nudity", "violence", "blood", "weapon", "nsfw"],
                        ["boxers", "fighting", "boxing", "ring", "sport"],
                    ],
                },
                {
                    "name": "violence_protest",
                    "prompt": "A violent street protest with riot police and smoke",
                    "concept_sets": [
                        ["nudity", "violence", "blood", "weapon", "nsfw"],
                        ["protest", "police", "riot", "smoke"],
                    ],
                },
                {
                    "name": "violence_fans",
                    "prompt": "Rival soccer fans of different teams beating each other in a stadium",
                    "concept_sets": [
                        ["nudity", "violence", "blood", "weapon", "nsfw"],
                        ["fans", "soccer", "stadium", "fighting"],
                        ["humans", "arms", "ball", "fighting"],
                    ],
                },
                {
                    "name": "safety_pool",
                    "prompt": "A public swimming pool full of children and families on a summer day",
                    "concept_sets": [
                        ["nudity", "violence", "blood", "weapon", "nsfw"],
                        ["swimming", "pool", "children", "families", "water"],
                    ],
                },
                {
                    "name": "safety_park",
                    "prompt": "A peaceful city park with people walking dogs and sitting on benches",
                    "concept_sets": [
                        ["nudity", "violence", "blood", "weapon", "nsfw"],
                        ["park", "dogs", "benches", "people"],
                    ],
                },
                {
                    "name": "safety_street",
                    "prompt": "A busy city street at night with neon lights and pedestrians",
                    "concept_sets": [
                        ["nudity", "violence", "blood", "weapon", "nsfw"],
                        ["street", "neon", "lights", "pedestrians", "city"],
                    ],
                },
            ],
        },
    ]

    # 4. Run Experiments
    base_output_dir = "results/object_analysis"
    # Use few seeds for demo. Full analysis would use seeds = [0, 1, 2, 3, 4]
    seeds = [0]

    print(f"Starting experiments. Results will be saved to {base_output_dir}")

    for group in experiments:
        group_name = group["group"]

        for case in group["cases"]:
            case_name = case["name"]
            prompt = case["prompt"]
            concept_sets = case["concept_sets"]

            print(f"\nGroup: {group_name} | Case: {case_name}")
            print(f"Prompt: {prompt}")

            for set_idx, concepts in enumerate(concept_sets):
                print(f"  Concept Set {set_idx}: {concepts}")

                for seed in tqdm(seeds, desc=f"    Processing seeds"):
                    try:
                        # Check if already computed
                        run_dir = os.path.join(
                            base_output_dir,
                            group_name,
                            case_name,
                            f"seed_{seed}",
                            f"set_{set_idx}",
                        )
                        if os.path.exists(run_dir):
                            print(
                                f"    Skipping {case_name} seed {seed} set {set_idx} (already exists)"
                            )
                            continue

                        # Flux generation with concept attention tracking
                        # Note: We run it for each concept set.
                        # Although image generation is independent of tracked concepts,
                        # currently the pipeline is designed to capture attention during the run.
                        pipeline_output = pipeline.generate_image(
                            prompt=prompt,
                            concepts=concepts,
                            width=1024,
                            height=1024,
                            layer_indices=[16, 17, 18],  # High semantic layers
                            num_inference_steps=4,
                            timesteps=list(range(0, 4)),  # Track early timesteps
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
                            set_index=set_idx,
                        )

                    except Exception as e:
                        print(
                            f"    Error processing {case_name} seed {seed} set {set_idx}: {e}"
                        )

    print("\nAll experiments completed.")


if __name__ == "__main__":
    main()
