import os
import json
import base64
import numpy as np
from pathlib import Path
from PIL import Image
from io import BytesIO
from tqdm import tqdm
from jinja2 import Environment, FileSystemLoader

CONCEPT_CLASSES = {
    # objects
    "hands": "object",
    "fingers": "object",
    "apple": "object",
    "person": "object",
    "fruit": "object",
    "sign": "object",
    "light": "object",
    "glass": "object",
    "water": "object",
    "table": "object",
    "wood": "object",
    "jar": "object",
    "marbles": "object",
    "objects": "object",
    "mirror": "object",
    "camera": "object",
    "frame": "object",
    "room": "object",
    "wool": "object",
    "yarn": "object",
    "bicycle": "object",
    "wheels": "object",
    "spokes": "object",
    "gears": "object",
    "chain": "object",
    "handlebar": "object",
    "springs": "object",
    "watch": "object",
    "brass": "object",
    "jewels": "object",
    "bottles": "object",
    "hair": "object",
    "strands": "object",
    "face": "object",
    "milk": "object",
    "strawberry": "object",
    "droplets": "object",
    "liquid": "object",
    "people": "object",
    "faces": "object",
    "group": "object",
    "friends": "object",
    "park": "object",
    "crowd": "object",
    "centipede": "object",
    "legs": "object",
    "eyes": "object",
    "leaf": "object",
    "insect": "object",
    "arthropod": "object",
    "microscope": "object",
    "lens": "object",
    "scientist": "object",
    "laboratory": "object",
    "objective": "object",
    "stage": "object",
    "instrument": "object",
    "cat": "object",
    "sofa": "object",
    "human": "object",
    "animal": "object",
    "furniture": "object",
    "creature": "object",
    "pet": "object",
    "module": "object",
    "hat": "object",
    "scarf": "object",
    "gloves": "object",
    "clothing": "object",
    "fire": "object",
    "ice": "object",
    "elephant": "object",
    "jungle": "object",
    "banana": "object",
    "grapes": "object",
    "zebra": "object",
    "savanna": "object",
    "bird": "object",
    "wings": "object",
    "beak": "object",
    "statue": "object",
    "keys": "object",
    "piano": "object",
    "rock": "object",
    "fur": "object",
    "stone": "object",
    "sphere": "object",
    "silk": "object",
    "cloth": "object",
    "slime": "object",
    "sandpaper": "object",
    "grains": "object",
    "chrome": "object",
    "metal": "object",
    "engine": "object",
    "sky": "object",
    "curtain": "object",
    "folds": "object",
    "fabric": "object",
    "earth": "object",
    "desert": "object",
    "soil": "object",
    "ground": "object",
    "bubbles": "object",
    "soap": "object",
    "smoke": "object",
    "vapor": "object",
    "silhouette": "object",
    "skin": "object",
    "python": "object",
    "serpent": "object",
    "scales": "object",
    "lava": "object",
    "fence": "object",
    "house": "object",
    "picket": "object",
    "brick": "object",
    "stack": "object",
    "books": "object",
    "candle": "object",
    "flame": "object",
    "box": "object",
    "pedestal": "object",
    "eye": "object",
    "keyhole": "object",
    "pyramid": "object",
    "row": "object",
    "ring": "object",
    "sand": "object",
    "beach": "object",
    "painting": "object",
    "wall": "object",
    "gallery": "object",
    "dogs": "object",
    "husky": "object",
    "animals": "object",
    "rim": "object",
    "wine": "object",
    "roots": "object",
    "tree": "object",
    "bridge": "object",
    "river": "object",
    "forest": "object",
    "astronaut": "object",
    "space": "object",
    "spider": "object",
    "circle": "object",
    "arms": "object",
    "tools": "object",
    "fountain": "object",
    "dragon": "object",
    "staircase": "object",
    "letters": "object",
    "landscape": "object",
    "coffee": "object",
    "cup": "object",
    "mountain": "object",
    "car": "object",
    "numbers": "object",
    "clock": "object",
    "weapon": "object",
    "sculpture": "object",
    "soldiers": "object",
    "trench": "object",
    "boxers": "object",
    "police": "object",
    "fans": "object",
    "soccer": "object",
    "stadium": "object",
    "pool": "object",
    "children": "object",
    "families": "object",
    "benches": "object",
    "pedestrians": "object",
    "city": "object",
    "street": "object",
    "horse": "object",
    "moon": "object",
    "text": "object",
    # colors
    "red": "color",
    "blue": "color",
    "pink": "color",
    "colors": "color",
    "yellow": "color",
    "green": "color",
    "white": "color",
    "orange": "color",
    "purple": "color",
    "black": "color",
    "violet": "color",
    "pastel": "color",
    "mint": "color",
    "lilac": "color",
    "rainbow": "color",
    "colorful": "color",
    "gold": "color",
    # textures
    "neon": "texture",
    "reflective": "texture",
    "translucent": "texture",
    "transparent": "texture",
    "crystalline": "texture",
    "fluffy": "texture",
    "rugged": "texture",
    "mossy": "texture",
    "smooth": "texture",
    "metallic": "texture",
    "shiny": "texture",
    "viscous": "texture",
    "sticky": "texture",
    "coarse": "texture",
    "gritty": "texture",
    "rough": "texture",
    "polished": "texture",
    "soft": "texture",
    "cracked": "texture",
    "dry": "texture",
    "irridescent": "texture",
    "thick": "texture",
    "frosted": "texture",
    "scaly": "texture",
    "glowing": "texture",
    "rusty": "texture",
    "knitted": "texture",
    "pattern": "texture",
    "weaving": "texture",
    "mechanical": "texture",
    "flowing": "texture",
    "splashing": "texture",
    "segmented": "texture",
    "velvet": "texture",
    "gradient": "texture",
    "shadows": "texture",
    "darkness": "texture",
    "surface": "texture",
    "cooling": "texture",
    "molten": "texture",
    "blurry": "texture",
    "thin": "texture",
    "gaseous": "texture",
    "heat": "texture",
    "material": "texture",
    # abstract
    "concept": "abstract",
    "typography": "abstract",
    "count": "abstract",
    "six": "abstract",
    "ten": "abstract",
    "movement": "abstract",
    "refraction": "abstract",
    "impact": "abstract",
    "surreal": "abstract",
    "minimalist": "abstract",
    "unusual": "abstract",
    "reversal": "abstract",
    "composition": "abstract",
    "anatomy": "abstract",
    "error": "abstract",
    "impossible": "abstract",
    "geometry": "abstract",
    "shape": "abstract",
    "logic": "abstract",
    "extra": "abstract",
    "hallucination": "abstract",
    "gravity": "abstract",
    "mismatch": "abstract",
    "mythical": "abstract",
    "blend": "abstract",
    "uncanny": "abstract",
    "loop": "abstract",
    "paradox": "abstract",
    "consept": "abstract",
    "spelling": "abstract",
    "scale": "abstract",
    "internal": "abstract",
    "time": "abstract",
    "structure": "abstract",
    "nudity": "abstract",
    "violence": "abstract",
    "blood": "abstract",
    "nsfw": "abstract",
    "sunbathing": "abstract",
    "renaissance": "abstract",
    "nude": "abstract",
    "male": "abstract",
    "battle": "abstract",
    "war": "abstract",
    "fighting": "abstract",
    "sport": "abstract",
    "riot": "abstract",
    "swimming": "abstract",
    "night": "abstract",
    "protest": "abstract",
    "seven": "abstract",
    "three": "abstract",
    "wrong": "abstract",
    "jumbled": "abstract",
    "lighting": "abstract",
    "spatial": "abstract",
    # spatial
    "reflection": "spatial",
    "overlapping": "spatial",
    "behind": "spatial",
    "in": "spatial",
    "front": "spatial",
    "of": "spatial",
    "distance": "spatial",
    "on": "spatial",
    "top": "spatial",
    "next": "spatial",
    "to": "spatial",
    "perched": "spatial",
    "inside": "spatial",
    "position": "spatial",
    "peering": "spatial",
    "through": "spatial",
    "occlusion": "spatial",
    "left": "spatial",
    "middle": "spatial",
    "right": "spatial",
    "floating": "spatial",
    "above": "spatial",
    "gap": "spatial",
    "buried": "spatial",
    "partially": "spatial",
    "depth": "spatial",
    "opposite": "spatial",
    "between": "spatial",
    "balanced": "spatial",
    "stability": "spatial",
    "around": "spatial",
    "growing": "spatial",
    "entanglement": "spatial",
    "over": "spatial",
    "span": "spatial",
    "riding": "spatial",
    "upwards": "spatial",
    "falling": "spatial",
    "dripping": "spatial",
    "standing": "spatial",
    "sitting": "spatial",
    "holding": "spatial",
    "bottom": "spatial",
}


def get_concept_class(concept):
    return CONCEPT_CLASSES.get(concept.lower(), "object")


def image_to_base64(img_path, max_size=(512, 512)):
    if not os.path.exists(img_path):
        return ""
    try:
        img = Image.open(img_path)
        img.thumbnail(max_size)
        buffered = BytesIO()
        img.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        return f"data:image/png;base64,{img_str}"
    except Exception:
        return ""


def get_metric_class(val):
    if val >= 0.7:
        return "metric-good"
    if val < 0.4:
        return "metric-poor"
    return "metric-fair"


def generate_gallery(results_dir, output_html, mode="relative"):
    results_dir = Path(results_dir)
    print(
        f"Generating hierarchical gallery with findings at {output_html} (mode={mode})..."
    )

    groups = {}
    all_cases = []

    for root, dirs, files in os.walk(results_dir):
        if Path(root).name.startswith("set_"):
            rel_root = Path(root).relative_to(results_dir)
            group_name = (
                str(rel_root.parents[1]) if len(rel_root.parents) > 1 else "Root"
            )
            if group_name not in groups:
                groups[group_name] = []
            groups[group_name].append(root)

            # Store flat list of cases for findings tab
            all_cases.append(root)

    # Initialize T5 Tokenizer and pre-compute concept tokens
    concept_to_tokens = {}
    unique_concepts = set()
    for case_path_str in all_cases:
        meta_path = Path(case_path_str) / "metadata.json"
        if meta_path.exists():
            try:
                with open(meta_path, "r") as f:
                    meta = json.load(f)
                unique_concepts.update(meta.get("concepts", []))
            except Exception:
                pass

    try:
        from transformers import T5Tokenizer
        import logging

        logging.getLogger("transformers").setLevel(logging.ERROR)
        t5_tokenizer = T5Tokenizer.from_pretrained("google/t5-v1_1-xxl")
        for c in unique_concepts:
            tokens = t5_tokenizer.tokenize(c)
            concept_to_tokens[c] = tokens
    except Exception as e:
        print(f"Warning: Could not load tokenizer for analysis: {e}")

    # Compute aggregate metrics for all concepts based on concept classes, split by token count
    def get_empty_metrics():
        return {
            "object": {"iou": [], "precision": [], "recall": []},
            "color": {"iou": [], "precision": [], "recall": []},
            "texture": {"iou": [], "precision": [], "recall": []},
            "abstract": {"iou": [], "precision": [], "recall": []},
            "spatial": {"iou": [], "precision": [], "recall": []},
        }

    single_token_metrics = get_empty_metrics()
    multi_token_metrics = get_empty_metrics()
    concept_iou_aggregates = {}

    # Store top performing concepts across all runs
    top_iou_candidates = []
    hallucination_candidates = []
    missed_localization_candidates = []

    for case_path_str in all_cases:
        meta_path = Path(case_path_str) / "metadata.json"
        if meta_path.exists():
            with open(meta_path, "r") as f:
                meta = json.load(f)
            sam_metrics_path = Path(case_path_str) / "sam_analysis" / "metrics.json"
            if sam_metrics_path.exists():
                try:
                    with open(sam_metrics_path, "r") as f:
                        all_sam_metrics = json.load(f)
                    for concept in meta.get("concepts", []):
                        if concept in all_sam_metrics:
                            c_class = get_concept_class(concept)
                            data = all_sam_metrics[concept]
                            iou_val = data.get("iou", 0)
                            prec_val = data.get("precision", 0)
                            rec_val = data.get("recall", 0)

                            tokens = concept_to_tokens.get(concept, [])
                            # Ignore special characters like \u2581 or "_" when counting real tokens
                            real_tokens = [
                                t for t in tokens if t not in ["\u2581", "_"]
                            ]
                            is_multi = len(real_tokens) > 1
                            target_metrics = (
                                multi_token_metrics
                                if is_multi
                                else single_token_metrics
                            )

                            target_metrics[c_class]["iou"].append(iou_val)
                            target_metrics[c_class]["precision"].append(prec_val)
                            target_metrics[c_class]["recall"].append(rec_val)

                            if concept not in concept_iou_aggregates:
                                concept_iou_aggregates[concept] = []
                            concept_iou_aggregates[concept].append(iou_val)

                            safe_concept = (
                                concept.replace(" ", "_")
                                .replace("'", "")
                                .replace('"', "")
                                .replace("/", "-")
                            )

                            # Log for top performances if valid IoU
                            if iou_val > 0:
                                top_iou_candidates.append(
                                    {
                                        "case_path": case_path_str,
                                        "prompt": meta.get("prompt", ""),
                                        "seed": Path(case_path_str).name.replace(
                                            "set_", ""
                                        ),
                                        "concept": concept,
                                        "iou": iou_val,
                                        "precision": prec_val,
                                        "recall": rec_val,
                                        "iou_rank_label": f"#{','.join(map(str, data.get('segment_indices', [])))} SAM Segs"
                                        if "segment_indices" in data
                                        else f"#{data.get('segment_index')} SAM Seg",
                                        "safe_concept": safe_concept,
                                    }
                                )

                            # Failure cases: Hallucination (low precision)
                            if prec_val < 0.15 and iou_val < 0.1:
                                hallucination_candidates.append(
                                    {
                                        "case_path": case_path_str,
                                        "prompt": meta.get("prompt", ""),
                                        "seed": Path(case_path_str).name.replace(
                                            "set_", ""
                                        ),
                                        "concept": concept,
                                        "iou": iou_val,
                                        "precision": prec_val,
                                        "recall": rec_val,
                                        "safe_concept": safe_concept,
                                    }
                                )
                            # Failure cases: Missed Localization (low recall)
                            if rec_val < 0.15 and iou_val < 0.1:
                                missed_localization_candidates.append(
                                    {
                                        "case_path": case_path_str,
                                        "prompt": meta.get("prompt", ""),
                                        "seed": Path(case_path_str).name.replace(
                                            "set_", ""
                                        ),
                                        "concept": concept,
                                        "iou": iou_val,
                                        "precision": prec_val,
                                        "recall": rec_val,
                                        "safe_concept": safe_concept,
                                    }
                                )
                except Exception:
                    pass

    # Compute averages helper
    def compute_averages(metrics_dict):
        averages = {}
        for c_class, metrics in metrics_dict.items():
            if len(metrics["iou"]) > 0:
                averages[c_class] = {
                    "iou": sum(metrics["iou"]) / len(metrics["iou"]),
                    "iou_std": float(np.std(metrics["iou"])),
                    "iou_min": float(np.min(metrics["iou"])),
                    "iou_max": float(np.max(metrics["iou"])),
                    "precision": sum(metrics["precision"]) / len(metrics["precision"]),
                    "prec_std": float(np.std(metrics["precision"])),
                    "prec_min": float(np.min(metrics["precision"])),
                    "prec_max": float(np.max(metrics["precision"])),
                    "recall": sum(metrics["recall"]) / len(metrics["recall"]),
                    "rec_std": float(np.std(metrics["recall"])),
                    "rec_min": float(np.min(metrics["recall"])),
                    "rec_max": float(np.max(metrics["recall"])),
                    "count": len(metrics["iou"]),
                }
            else:
                averages[c_class] = {
                    "iou": 0,
                    "iou_std": 0,
                    "iou_min": 0,
                    "iou_max": 0,
                    "precision": 0,
                    "prec_std": 0,
                    "prec_min": 0,
                    "prec_max": 0,
                    "recall": 0,
                    "rec_std": 0,
                    "rec_min": 0,
                    "rec_max": 0,
                    "count": 0,
                }
        return averages

    single_token_averages = compute_averages(single_token_metrics)
    multi_token_averages = compute_averages(multi_token_metrics)

    umap_path = results_dir / "concept_umap.json"
    umap_js_data = []
    if umap_path.exists():
        try:
            with open(umap_path, "r") as f:
                umap_coords = json.load(f)

            concept_avg_iou = {
                c: sum(v) / len(v)
                for c, v in concept_iou_aggregates.items()
                if len(v) > 0
            }

            for concept, coords in umap_coords.items():
                c_class = get_concept_class(concept)
                umap_js_data.append(
                    {
                        "x": coords[0],
                        "y": coords[1],
                        "label": concept,
                        "category": c_class,
                        "iou": concept_avg_iou.get(concept, 0.0),
                    }
                )
        except Exception as e:
            print(f"Warning: Could not load UMAP data: {e}")

    def get_image_src(path):
        if not Path(path).exists():
            return ""
        if mode == "base64":
            return image_to_base64(path)
        elif mode == "absolute":
            return f"file://{Path(path).absolute()}"
        else:
            html_parent = Path(output_html).parent
            return os.path.relpath(path, html_parent)

    sorted_group_names = sorted(groups.keys())

    # ── Prepare Top Candidate data ──
    top_iou_candidates.sort(key=lambda x: x["iou"], reverse=True)
    top_prec_candidates = sorted(
        top_iou_candidates, key=lambda x: x["precision"], reverse=True
    )
    top_rec_candidates = sorted(
        top_iou_candidates, key=lambda x: x["recall"], reverse=True
    )

    metric_label_map = {"iou": "IoU", "precision": "Prec", "recall": "Rec"}

    def enrich_candidate(cand):
        """Compute image sources and metric classes for a candidate dict."""
        case_path = Path(cand["case_path"])
        orig_img = case_path / "image.png"
        masked_img = case_path / f"masked_image_{cand['safe_concept']}.png"
        sam_img = (
            case_path
            / "sam_analysis"
            / f"matched_masked_image_{cand['safe_concept']}.png"
        )
        coverage_img = case_path / "sam_analysis" / "debug_coverage_gaps.png"

        if not masked_img.exists():
            masked_img = case_path / f"upscaled_heatmap_{cand['safe_concept']}.png"
            if not masked_img.exists():
                masked_img = case_path / f"heatmap_{cand['safe_concept']}.png"

        coverage_percent = 0.0
        summary_path = case_path / "sam_analysis" / "segments_summary.json"
        if summary_path.exists():
            try:
                with open(summary_path, "r") as f:
                    summary_data = json.load(f)
                coverage_percent = summary_data.get("overall_coverage_percent", 0.0)
            except Exception:
                pass

        tokens = concept_to_tokens.get(cand["concept"], [])
        cand["tokens_json"] = json.dumps(tokens)
        cand["orig_img_src"] = get_image_src(orig_img)
        cand["masked_img_src"] = get_image_src(masked_img)
        cand["sam_img_src"] = get_image_src(sam_img)
        cand["coverage_img_src"] = get_image_src(coverage_img)
        cand["coverage_percent"] = coverage_percent
        cand["iou_class"] = get_metric_class(cand["iou"])
        cand["prec_class"] = get_metric_class(cand["precision"])
        cand["rec_class"] = get_metric_class(cand["recall"])
        return cand

    top_candidate_sections = [
        {
            "candidates": [enrich_candidate(c) for c in top_iou_candidates[:15]],
            "metric_name": "Intersection over Union (IoU)",
            "metric_key": "iou",
            "m_label": metric_label_map["iou"],
            "color_hex": "#58a6ff",
        },
        {
            "candidates": [enrich_candidate(c) for c in top_prec_candidates[:15]],
            "metric_name": "Precision",
            "metric_key": "precision",
            "m_label": metric_label_map["precision"],
            "color_hex": "#3fb950",
        },
        {
            "candidates": [enrich_candidate(c) for c in top_rec_candidates[:15]],
            "metric_name": "Recall",
            "metric_key": "recall",
            "m_label": metric_label_map["recall"],
            "color_hex": "#d29922",
        },
    ]

    # ── Prepare tokenization data ──
    concept_tokens_list = []
    for c, tokens in concept_to_tokens.items():
        real_tokens = [t for t in tokens if t not in ["\u2581", "_"]]
        concept_tokens_list.append(
            {"concept": c, "tokens": str(tokens), "count": len(real_tokens)}
        )
    concept_tokens_list.sort(key=lambda x: x["count"], reverse=True)
    multi_tokens = [ct for ct in concept_tokens_list if ct["count"] > 1]

    # ── Prepare findings grid items ──
    highlighted_keys = [
        "blue_cat",
        "dog_human_face",
        "violence_fans",
        "purple_banana",
        "violence_war",
    ]
    findings_grid_items = []
    findings_grids_dir = Path("experiments/results_heatmap_grids")
    if findings_grids_dir.exists():
        for grid_file in sorted(findings_grids_dir.glob("*.png")):
            filename = grid_file.name
            if any(key in filename for key in highlighted_keys):
                clean_name = filename.replace(".png", "")
                for prefix in [
                    "attribute_color_",
                    "attribute_spatial_",
                    "attribute_texture_",
                    "attribute_",
                    "baseline_",
                    "failure_cases_",
                    "safety_filter_",
                ]:
                    if clean_name.startswith(prefix):
                        clean_name = clean_name[len(prefix) :]
                clean_name = clean_name.split("_seed_")[0]
                search_term = clean_name.replace("_", " ").title()
                findings_grid_items.append(
                    {
                        "filename": filename,
                        "filename_lower": filename.lower(),
                        "search_term": search_term,
                        "search_term_lower": search_term.lower(),
                        "img_src": get_image_src(grid_file),
                    }
                )

    # ── Prepare gallery groups ──
    gallery_groups = []
    for idx, group_name in enumerate(
        tqdm(sorted_group_names, desc="Generating Gallery Sections")
    ):
        set_paths_in_group = sorted(groups[group_name])

        cases_in_group = {}
        for set_path_str in set_paths_in_group:
            meta_path = Path(set_path_str) / "metadata.json"
            if meta_path.exists():
                with open(meta_path, "r") as f:
                    meta = json.load(f)
                case_id = (
                    Path(set_path_str).parent.parent.name
                    + "_"
                    + Path(set_path_str).parent.name
                )
                if case_id not in cases_in_group:
                    cases_in_group[case_id] = []
                cases_in_group[case_id].append((set_path_str, meta))

        cases = []
        for case_id, sets_for_case in cases_in_group.items():
            if not sets_for_case:
                continue
            first_set_str, first_meta = sets_for_case[0]
            first_set_path = Path(first_set_str)
            seed = first_set_path.parent.name
            case_name = first_set_path.parent.parent.name
            image_path = first_set_path / "image.png"
            coverage_img_path = (
                first_set_path / "sam_analysis" / "debug_coverage_gaps.png"
            )

            coverage_percent = 0.0
            summary_path = first_set_path / "sam_analysis" / "segments_summary.json"
            if summary_path.exists():
                try:
                    with open(summary_path, "r") as f:
                        summary_data = json.load(f)
                    coverage_percent = summary_data.get("overall_coverage_percent", 0.0)
                except Exception:
                    pass

            prompt = first_meta.get("prompt", "")
            comments = [
                m.get("comment", "") for _, m in sets_for_case if m.get("comment")
            ]

            all_concepts_for_search = set()
            all_categories = set()
            for _, m in sets_for_case:
                concepts = m.get("concepts", [])
                all_concepts_for_search.update(concepts)
                for c in concepts:
                    all_categories.add(get_concept_class(c))

            # Build sets data
            sets_data = []
            for set_path_str_inner, meta_inner in sets_for_case:
                set_path = Path(set_path_str_inner)
                set_idx = set_path.name
                concepts_list = meta_inner.get("concepts", [])
                concepts_str = ", ".join(f"'{c}'" for c in concepts_list)

                # Load SAM metrics for this set
                sam_analysis_dir = set_path / "sam_analysis"
                sam_metrics_path = sam_analysis_dir / "metrics.json"
                all_sam_metrics = {}
                if sam_metrics_path.exists():
                    try:
                        with open(sam_metrics_path, "r") as f:
                            all_sam_metrics = json.load(f)
                    except Exception:
                        pass

                concept_items = []
                for concept in concepts_list:
                    tokens = concept_to_tokens.get(concept, [])
                    safe_concept = (
                        concept.replace(" ", "_")
                        .replace("'", "")
                        .replace('"', "")
                        .replace("/", "-")
                    )
                    masked_path = set_path / f"masked_image_{safe_concept}.png"
                    upscaled_path = set_path / f"upscaled_heatmap_{safe_concept}.png"
                    orig_heatmap_path = set_path / f"heatmap_{safe_concept}.png"
                    sam_masked_path = (
                        sam_analysis_dir / f"matched_masked_image_{safe_concept}.png"
                    )

                    primary_display = (
                        masked_path
                        if masked_path.exists()
                        else (
                            upscaled_path
                            if upscaled_path.exists()
                            else orig_heatmap_path
                        )
                    )

                    sam_data_raw = all_sam_metrics.get(concept)
                    sam_data = None
                    sam_segment_label = ""
                    if sam_data_raw:
                        iou = sam_data_raw.get("iou", 0)
                        precision = sam_data_raw.get("precision", 0)
                        recall = sam_data_raw.get("recall", 0)
                        if "segment_indices" in sam_data_raw:
                            sam_segment_label = ",".join(
                                map(str, sam_data_raw["segment_indices"])
                            )
                        else:
                            sam_segment_label = str(
                                sam_data_raw.get("segment_index", "")
                            )
                        sam_data = {
                            "iou": iou,
                            "precision": precision,
                            "recall": recall,
                            "iou_class": get_metric_class(iou),
                            "precision_class": get_metric_class(precision),
                            "recall_class": get_metric_class(recall),
                        }

                    concept_items.append(
                        {
                            "name": concept,
                            "tokens_json": json.dumps(tokens),
                            "primary_src": get_image_src(primary_display),
                            "upscaled_src": get_image_src(upscaled_path),
                            "sam_data": sam_data,
                            "sam_segment_label": sam_segment_label,
                            "sam_masked_src": get_image_src(sam_masked_path)
                            if sam_masked_path.exists()
                            else "",
                        }
                    )

                sets_data.append(
                    {
                        "set_idx": set_idx,
                        "concepts_str": concepts_str,
                        "concepts": concept_items,
                    }
                )

            cases.append(
                {
                    "case_name": case_name,
                    "case_name_lower": case_name.lower(),
                    "seed": seed,
                    "prompt": prompt,
                    "image_src": get_image_src(image_path),
                    "coverage_img_src": get_image_src(coverage_img_path),
                    "coverage_percent": coverage_percent,
                    "search_text": f"{prompt.lower()} {case_name.lower()} {' '.join(all_concepts_for_search).lower()}",
                    "categories_str": " ".join(all_categories).lower(),
                    "comments": comments,
                    "sets": sets_data,
                }
            )

        gallery_groups.append({"idx": idx, "name": group_name, "cases": cases})

    # ── Prepare failure analysis data ──
    hallucination_candidates.sort(key=lambda x: x["iou"])
    missed_localization_candidates.sort(key=lambda x: x["iou"])

    for cand in hallucination_candidates[:10] + missed_localization_candidates[:10]:
        enrich_candidate(cand)

    # ── Prepare grids data ──
    grid_items = []
    grids_dir = Path("experiments/results_heatmap_grids")
    if grids_dir.exists():
        for grid_file in sorted(grids_dir.glob("*.png")):
            filename = grid_file.name
            clean_name = filename.replace(".png", "")
            for prefix in [
                "attribute_color_",
                "attribute_spatial_",
                "attribute_texture_",
                "attribute_",
                "baseline_",
                "failure_cases_",
                "safety_filter_",
            ]:
                if clean_name.startswith(prefix):
                    clean_name = clean_name[len(prefix) :]
            clean_name = clean_name.split("_seed_")[0]
            search_term = clean_name.replace("_", " ").title()
            grid_items.append(
                {
                    "filename": filename,
                    "filename_lower": filename.lower(),
                    "search_term_lower": search_term.lower(),
                    "img_src": get_image_src(grid_file),
                }
            )

    # ── Render template ──
    template_dir = Path(__file__).parent / "templates"
    env = Environment(
        loader=FileSystemLoader(str(template_dir)),
        autoescape=False,
        keep_trailing_newline=True,
    )
    template = env.get_template("base.html")

    html_content = template.render(
        # Findings tab
        top_candidate_sections=top_candidate_sections,
        multi_tokens=multi_tokens,
        findings_grid_items=findings_grid_items,
        single_token_averages_json=json.dumps(single_token_averages),
        multi_token_averages_json=json.dumps(multi_token_averages),
        umap_js_data_json=json.dumps(umap_js_data),
        # Gallery tab
        gallery_groups=gallery_groups,
        # Failure tab
        hallucination_candidates=hallucination_candidates[:10],
        missed_localization_candidates=missed_localization_candidates[:10],
        # Grids tab
        grid_items=grid_items,
        # Tokenization tab
        concept_tokens_list=concept_tokens_list,
    )

    with open(output_html, "w", encoding="utf-8") as f:
        f.write(html_content)


if __name__ == "__main__":
    results_dir = "results/object_analysis"
    rel_html = os.path.join(results_dir, "evaluation_gallery.html")
    generate_gallery(results_dir, rel_html, mode="relative")

    full_html = os.path.join(results_dir, "evaluation_gallery_full.html")
    generate_gallery(results_dir, full_html, mode="base64")
