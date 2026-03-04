import os
import json
import base64
import numpy as np
from pathlib import Path
from PIL import Image
from io import BytesIO
from tqdm import tqdm

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

    # CSS and Layout Base
    html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Object Analysis Evaluation</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        html {{ scroll-behavior: smooth; }}
        body {{ font-family: 'Inter', system-ui, -apple-system, sans-serif; background: #0f1115; color: #e1e4e8; margin: 0; padding: 0; line-height: 1.6; display: flex; flex-direction: column; height: 100vh; }}
        
        .header-nav {{ background: #161b22; padding: 15px 30px; border-bottom: 1px solid #30363d; display: flex; justify-content: space-between; align-items: center; }}
        .header-nav h1 {{ margin: 0; color: #58a6ff; font-size: 1.5rem; }}
        
        .tabs {{ display: flex; gap: 20px; }}
        .tab-btn {{ background: transparent; border: none; color: #8b949e; font-size: 1.1rem; font-weight: bold; cursor: pointer; padding: 10px 15px; border-bottom: 3px solid transparent; transition: all 0.2s; }}
        .tab-btn:hover {{ color: #c9d1d9; }}
        .tab-btn.active {{ color: #58a6ff; border-bottom-color: #58a6ff; }}

        .main-container {{ display: flex; flex: 1; overflow: hidden; }}
        
        /* Table of Contents / Sidebar */
        .sidebar {{ width: 250px; background: #0d1117; border-right: 1px solid #30363d; overflow-y: auto; padding: 20px; display: none; }}
        .sidebar h3 {{ color: #8b949e; margin-top: 0; font-size: 0.9rem; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 15px; }}
        .toc-list {{ list-style: none; padding: 0; margin: 0; }}
        .toc-item {{ margin-bottom: 10px; }}
        .toc-link {{ color: #c9d1d9; text-decoration: none; font-size: 0.95rem; display: block; padding: 8px; border-radius: 6px; transition: background 0.2s; }}
        .toc-link:hover {{ background: #21262d; color: #58a6ff; }}
        
        .content-area {{ flex: 1; overflow-y: auto; padding: 30px; }}
        
        .tab-content {{ display: none; max-width: 1200px; margin: 0 auto; }}
        .tab-content.active {{ display: block; }}

        /* Finding and Gallery Cards */
        .case-block {{ background: #0d1117; border: 1px solid #30363d; border-radius: 8px; padding: 20px; margin-bottom: 25px; }}
        .case-info {{ margin-bottom: 20px; border-left: 4px solid #58a6ff; padding-left: 15px; }}
        .case-title {{ font-size: 1.2rem; font-weight: bold; color: #c9d1d9; font-family: monospace; }}
        .prompt {{ font-style: italic; color: #8b949e; margin: 8px 0; font-size: 0.95rem; }}
        
        .image-grid {{ display: grid; grid-template-columns: 1fr 2fr; gap: 30px; align-items: start; }}
        .main-image-container {{ text-align: center; }}
        .main-image-container img {{ width: 100%; max-width: 400px; border-radius: 8px; box-shadow: 0 4px 12px rgba(0,0,0,0.5); border: 1px solid #30363d; }}
        
        .heatmap-grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(300px, 1fr)); gap: 15px; }}
        .concept-block {{ background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 15px; display: flex; flex-direction: column; gap: 10px; }}
        .concept-label {{ font-size: 1.1rem; font-weight: bold; color: #58a6ff; margin-bottom: 5px; border-bottom: 1px solid #30363d; padding-bottom: 5px; }}
        
        .viz-grid {{ display: flex; flex-direction: column; gap: 10px; }}
        .viz-item {{ text-align: center; background: #0d1117; border: 1px solid #30363d; border-radius: 6px; padding: 10px; }}
        .viz-item img {{ max-width: 100%; height: auto; border-radius: 4px; border: 1px solid #21262d; transition: transform 0.2s; }}
        .viz-item img:hover {{ transform: scale(1.02); border-color: #58a6ff; }}
        .viz-label {{ font-size: 0.8rem; color: #8b949e; margin-top: 5px; }}
        
        .sam-results {{ margin-top: 15px; border: 1px solid #30363d; border-radius: 6px; overflow: hidden; font-size: 0.8rem; }}
        .sam-header {{ background: #161b22; padding: 8px; font-weight: bold; border-bottom: 1px solid #30363d; display: flex; justify-content: space-between; }}
        .sam-metrics {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(100px, 1fr)); gap: 8px; padding: 10px; background: #0d1117; }}
        .metric-item {{ background: #161b22; padding: 8px; border-radius: 4px; border: 1px solid #21262d; text-align: center; }}
        .metric-label {{ font-size: 0.7rem; color: #8b949e; margin-bottom: 2px; text-transform: uppercase; }}
        .metric-val {{ font-weight: bold; font-family: monospace; font-size: 0.9rem; }}
        
        .metric-good {{ color: #3fb950; }}
        .metric-fair {{ color: #d29922; }}
        .metric-poor {{ color: #f85149; }}

        .comment-section {{ margin-top: 25px; padding-top: 20px; border-top: 1px solid #30363d; background: rgba(88, 166, 255, 0.05); padding: 15px; border-radius: 8px; }}
        .comment-label {{ font-weight: bold; color: #58a6ff; margin-bottom: 10px; display: block; font-size: 1rem; }}
        .viz-item summary {{ background: #161b22; border: 1px solid #30363d; border-radius: 4px; padding: 8px; font-size: 0.9rem; color: #c9d1d9; }}
        .viz-item summary:hover {{ background: #30363d; }}
        .heatmap-toggles {{ display: flex; flex-direction: column; gap: 10px; margin-top: 10px; }}
        .heatmap-toggles > div {{ background: #0d1117; border: 1px solid #21262d; border-radius: 4px; padding: 8px; }}

        /* Top Search Bar */
        .search-container {{ padding: 20px 30px; border-bottom: 1px solid #30363d; margin: -30px -30px 30px -30px; background: rgba(15, 17, 21, 0.95); backdrop-filter: blur(8px); position: sticky; top: -30px; z-index: 100; }}
        .search-input {{ width: 100%; max-width: 600px; padding: 12px 20px; border-radius: 8px; border: 1px solid #30363d; background: #161b22; color: #e1e4e8; font-size: 1rem; outline: none; transition: border-color 0.2s; }}
        .search-input:focus {{ border-color: #58a6ff; box-shadow: 0 0 0 3px rgba(88, 166, 255, 0.1); }}

        .category-filters {{ display: flex; gap: 10px; margin-top: 15px; flex-wrap: wrap; }}
        .cat-btn {{ background: #161b22; border: 1px solid #30363d; color: #8b949e; padding: 6px 12px; border-radius: 20px; cursor: pointer; font-size: 0.85rem; transition: all 0.2s; }}
        .cat-btn:hover {{ border-color: #8b949e; color: #c9d1d9; }}
        .cat-btn.active {{ background: #58a6ff33; border-color: #58a6ff; color: #58a6ff; }}
        
        .findings-grid {{ display: grid; grid-template-columns: 1fr; gap: 30px; }}
        
        /* Groups for Gallery */
        .group-section {{ margin-bottom: 50px; padding-top: 20px; }}
        .group-section h2 {{ border-bottom: 2px solid #30363d; padding-bottom: 10px; color: #fff; }}
        
        .sets-details {{ background: transparent; border: 1px solid #30363d; border-radius: 8px; margin-top: 20px; overflow: hidden; }}
        .sets-summary {{ padding: 12px 15px; cursor: pointer; font-weight: 600; font-size: 1rem; background: #21262d; outline: none; display: flex; align-items: center; color: #e1e4e8; }}
        .sets-summary:hover {{ background: #30363d; }}
        .sets-summary:after {{ content: '▸'; margin-left: auto; transition: transform 0.2s; }}
        .sets-details[open] .sets-summary:after {{ transform: rotate(90deg); }}
        
        .set-block {{ border-top: 1px solid #30363d; padding: 20px; background: #0d1117; }}
        .set-title {{ color: #8b949e; font-size: 0.9rem; margin-bottom: 15px; font-weight: bold; text-transform: uppercase; letter-spacing: 1px; }}
    </style>
</head>
<body>
    <div class="header-nav">
        <h1>Concept Attention Analysis</h1>
        <div class="tabs">
            <button class="tab-btn active" onclick="switchTab('findings')">Findings</button>
            <button class="tab-btn" onclick="switchTab('gallery')">Gallery</button>
            <button class="tab-btn" onclick="switchTab('grids')">Grids</button>
            <button class="tab-btn" onclick="switchTab('tokenization')">Tokenization</button>
        </div>
    </div>
    
    <div class="main-container">
        <!-- Sidebar ONLY visible for Gallery -->
        <div class="sidebar" id="gallery-sidebar">
            <h3>Table of Contents</h3>
            <ul class="toc-list">
    """

    for idx, group_name in enumerate(sorted_group_names):
        html_content += f'<li class="toc-item"><a href="#group-{idx}" class="toc-link">{group_name}</a></li>'

    html_content += """
            </ul>
        </div>
        
        <div class="content-area">
            
            <!-- FINDINGS TAB -->
            <div id="findings" class="tab-content active">
            
                <!-- NEW EXPLANATION BLOCK -->
                <div class="explanation-block" style="margin-bottom: 30px; padding: 20px; background: rgba(88, 166, 255, 0.05); border: 1px solid #30363d; border-radius: 8px; border-left: 4px solid #58a6ff;">
                    <h2 style="margin-top: 0; color: #c9d1d9; font-size: 1.3rem;">Welcome to the Concept Analysis Evaluation Dashboard</h2>
                    <p style="color: #8b949e; margin-bottom: 15px;">
                        This webpage presents a comprehensive evaluation of the <b>Concept Attention</b> mechanism, visualizing how well it localizes various concepts—ranging from concrete objects and colors to textures, spatial relations, and abstract ideas—within generated images.
                        The 1024x1024 pixel images are generated using the Flux1.schnell model in 4 inference steps, and the concept heatmaps are extracted from layers 16, 17, and 18 of the double stream attention blocks.
                    </p>
                    <div style="background: rgba(210, 153, 34, 0.1); border: 1px solid #d29922; border-radius: 6px; padding: 12px; margin-bottom: 15px; color: #d29922; font-size: 0.9rem;">
                        <strong>Technical Note:</strong> For concepts consisting of multiple text tokens (as determined by the T5 tokenizer), <strong>only the first token's attention map</strong> is currently used for localization. 
                        Note that the special character <code style="background: #161b22; padding: 2px 4px; border-radius: 3px;">\u2581</code> (often rendered as <code style="background: #161b22; padding: 2px 4px; border-radius: 3px;">_</code>) is <strong>not considered a real token</strong> and is ignored during count calculations.
                    </div>
                    <ul style="color: #8b949e; margin-bottom: 15px; padding-left: 20px;">
                        <li style="margin-bottom: 8px;"><strong style="color: #c9d1d9;">Findings:</strong> Displays a high-level quantitative overview followed by key highlighted cases. Metrics (IoU, Precision, Recall) are computed by comparing the Concept Attention heatmaps against masks generated by the Segment Anything Model (SAM), acting as an automated ground truth.</li>
                        <li style="margin-bottom: 8px;"><strong style="color: #c9d1d9;">Gallery:</strong> A hierarchical browser to explore all generated images, prompts, and seeds. Expand concept sets to compare primary masked images, raw attention maps, and SAM segments side-by-side.</li>
                        <li style="margin-bottom: 8px;"><strong style="color: #c9d1d9;">Grids:</strong> A dense visual layout designed for qualitative assessment of attention heatmaps across different experimental runs.</li>
                        <li style="margin-bottom: 8px;"><strong style="color: #c9d1d9;">Tokenization:</strong> Illustrates how each concept is split into text tokens by the T5 encoder. This is critical for understanding localization, as multi-token concepts may exhibit different attention behaviors compared to single-token ones.</li>
                    </ul>
                    <p style="color: #8b949e; font-size: 0.9rem; margin-bottom: 0;">
                        <i>Tip: Use the tabs above to navigate between views, and expand the "Concept Sets" or "View Alternate Heatmaps" details in the cases below for deeper visual analysis.</i>
                    </p>
                </div>

                <div class="statistics-section" style="margin-bottom: 40px; padding: 20px; background: #0d1117; border: 1px solid #30363d; border-radius: 8px;">
                    <h3 style="margin-top:0; color:#58a6ff;">1. Single-Token Concept Statistics</h3>
                    <p style="color: #8b949e; margin-top: 0;">Metrics for concepts that map to exactly one T5 token.</p>
                    <div style="display: flex; flex-wrap: wrap; gap: 20px; margin-bottom: 30px;">
                        <div style="flex: 1; min-width: 400px; background: #161b22; padding: 15px; border: 1px solid #30363d; border-radius: 8px;">
                            <canvas id="singleMetricsChart"></canvas>
                        </div>
                        <div style="flex: 1; min-width: 400px; background: #161b22; padding: 15px; border: 1px solid #30363d; border-radius: 8px;">
                            <canvas id="singleCountsChart"></canvas>
                        </div>
                    </div>

                    <h3 style="margin-top:0; color:#d29922;">2. Multi-Token Concept Statistics</h3>
                    <p style="color: #8b949e; margin-top: 0;">Metrics for concepts that are split into multiple T5 tokens (only the first token is currently used).</p>
                    <div style="display: flex; flex-wrap: wrap; gap: 20px;">
                        <div style="flex: 1; min-width: 400px; background: #161b22; padding: 15px; border: 1px solid #30363d; border-radius: 8px;">
                            <canvas id="multiMetricsChart"></canvas>
                        </div>
                        <div style="flex: 1; min-width: 400px; background: #161b22; padding: 15px; border: 1px solid #30363d; border-radius: 8px;">
                            <canvas id="multiCountsChart"></canvas>
                        </div>
                    </div>
                </div>
    """

    # Prepare Top Performing Concepts sections
    top_iou_candidates.sort(key=lambda x: x["iou"], reverse=True)
    top_prec_candidates = sorted(
        top_iou_candidates, key=lambda x: x["precision"], reverse=True
    )
    top_rec_candidates = sorted(
        top_iou_candidates, key=lambda x: x["recall"], reverse=True
    )

    def render_top_candidates(candidates, metric_name, metric_key, color_hex):
        display_candidates = candidates[:15]  # Top 15
        if not display_candidates:
            return ""

        metric_label_map = {"iou": "IoU", "precision": "Prec", "recall": "Rec"}
        m_label = metric_label_map.get(metric_key, metric_key.title())

        # Determine metric class for background/border coloring
        def get_local_metric_class(v):
            if v >= 0.7:
                return "metric-good"
            if v < 0.4:
                return "metric-poor"
            return "metric-fair"

        section_html = f"""
                <details style="margin-bottom: 40px; background: #0d1117; border: 1px solid #30363d; border-radius: 8px; padding: 15px;">
                    <summary style="cursor: pointer; padding: 10px; font-weight: bold; font-size: 1.2rem; color: {color_hex}; outline: none;">
                        Top Performing Concepts Ordered by {metric_name}
                    </summary>
                    <div style="margin-top: 20px; display: flex; flex-direction: column; gap: 30px;">
        """

        for cand in display_candidates:
            case_path = Path(cand["case_path"])
            orig_img = case_path / "image.png"
            masked_img = case_path / f"masked_image_{cand['safe_concept']}.png"
            sam_img = (
                case_path
                / "sam_analysis"
                / f"matched_masked_image_{cand['safe_concept']}.png"
            )
            coverage_img = case_path / "sam_analysis" / "debug_coverage_gaps.png"

            coverage_percent = 0.0
            summary_path = case_path / "sam_analysis" / "segments_summary.json"
            if summary_path.exists():
                try:
                    with open(summary_path, "r") as f:
                        summary_data = json.load(f)
                    coverage_percent = summary_data.get("overall_coverage_percent", 0.0)
                except Exception:
                    pass

            # Fallback if masked image not present
            if not masked_img.exists():
                masked_img = case_path / f"upscaled_heatmap_{cand['safe_concept']}.png"
                if not masked_img.exists():
                    masked_img = case_path / f"heatmap_{cand['safe_concept']}.png"

            tokens = concept_to_tokens.get(cand["concept"], [])
            tokens_json = json.dumps(tokens)

            section_html += f"""
                    <div class="case-block" style="border-left: 4px solid {color_hex}; margin-bottom: 0;">
                        <div style="display: flex; justify-content: space-between; align-items: start; margin-bottom: 20px; flex-wrap: wrap; gap: 15px;">
                            <div>
                                <div class="case-title" style="color: {color_hex}; display: inline-block; margin-right: 15px; font-size: 1.4rem;">{cand["concept"]}</div>
                                <div class="metric-item" style="border-color: {color_hex}; background: {color_hex}1a; display: inline-block; vertical-align: bottom;">
                                    <span style="color: {color_hex}; font-weight: bold; font-family: monospace; font-size: 1.1rem;">{m_label}: {cand[metric_key]:.2%}</span>
                                </div>
                                <div style="color: #8b949e; font-family: monospace; margin-top: 8px;">{cand["seed"]} | "{cand["prompt"]}"</div>
                                <div style="color: #8b949e; font-family: monospace; font-size: 0.8rem; margin-top: 4px;">Tokens: {tokens_json}</div>
                            </div>
                            <div style="display: flex; gap: 10px;">
                                <div class="metric-item" title="IoU"><span class="metric-label">IoU</span><br><span class="metric-val {get_local_metric_class(cand["iou"])}">{cand["iou"]:.2%}</span></div>
                                <div class="metric-item" title="Precision"><span class="metric-label">Prec</span><br><span class="metric-val {get_local_metric_class(cand["precision"])}">{cand["precision"]:.2%}</span></div>
                                <div class="metric-item" title="Recall"><span class="metric-label">Rec</span><br><span class="metric-val {get_local_metric_class(cand["recall"])}">{cand["recall"]:.2%}</span></div>
                            </div>
                        </div>
                        
                        <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 15px;">
                            <div style="text-align: center; background: #161b22; padding: 10px; border-radius: 6px; border: 1px solid #30363d;">
                                <img src="{get_image_src(orig_img)}" style="width: 100%; border-radius: 4px; margin-bottom: 8px;" loading="lazy">
                                <div style="color: #8b949e; font-size: 0.85rem;">Generated Image</div>
                            </div>
                            <div style="text-align: center; background: #161b22; padding: 10px; border-radius: 6px; border: 1px solid #30363d;">
                                <img src="{get_image_src(masked_img)}" style="width: 100%; border-radius: 4px; margin-bottom: 8px;" loading="lazy">
                                <div style="color: #8b949e; font-size: 0.85rem;">Concept Attention</div>
                            </div>
                            <div style="text-align: center; background: #161b22; padding: 10px; border-radius: 6px; border: 1px solid #30363d;">
                                <img src="{get_image_src(sam_img)}" style="width: 100%; border-radius: 4px; margin-bottom: 8px;" loading="lazy">
                                <div style="color: #8b949e; font-size: 0.85rem;">SAM Segment</div>
                            </div>
                            <div style="text-align: center; background: #161b22; padding: 10px; border-radius: 6px; border: 1px solid #30363d;">
                                <img src="{get_image_src(coverage_img)}" style="width: 100%; border-radius: 4px; margin-bottom: 8px;" loading="lazy">
                                <div style="color: #8b949e; font-size: 0.85rem;">SAM Coverage ({coverage_percent:.1f}%)</div>
                            </div>
                        </div>
                    </div>
            """
        section_html += """
                    </div>
                </details>
        """
        return section_html

    # Render metrics sections
    html_content += """
                <h2 style="margin-top:0; color:#bc8cff;">Concept Similarity Map (UMAP)</h2>
                <div style="display: flex; justify-content: space-between; align-items: flex-end; margin-bottom: 20px;">
                    <p style="color: #8b949e; margin: 0;">Proximity indicates semantic similarity in T5 embedding space.</p>
                    <div style="display: flex; align-items: center; gap: 10px;">
                        <span style="color: #8b949e; font-size: 0.9rem;">Color by:</span>
                        <select id="umapColorToggle" style="background: #0d1117; color: #c9d1d9; border: 1px solid #30363d; padding: 5px 10px; border-radius: 4px; outline: none; cursor: pointer;">
                            <option value="category">Category</option>
                            <option value="iou">IoU Gradient</option>
                        </select>
                    </div>
                </div>
                <div style="background: #161b22; padding: 25px; border: 1px solid #30363d; border-radius: 8px; margin-bottom: 40px; height: 850px;">
                    <canvas id="umapChart"></canvas>
                </div>

                <h2 style="margin-top:0;">Top Performing Concepts per Metric</h2>
                <p style="color: #8b949e; margin-bottom: 30px;">This section identifies the best localized concepts across the entire dataset, ranked by specific quantitative metrics. Expand a section below to explore the top 15 candidates.</p>
    """
    html_content += render_top_candidates(
        top_iou_candidates, "Intersection over Union (IoU)", "iou", "#58a6ff"
    )
    html_content += render_top_candidates(
        top_prec_candidates, "Precision", "precision", "#3fb950"
    )
    html_content += render_top_candidates(
        top_rec_candidates, "Recall", "recall", "#d29922"
    )

    # Prepare Tokenization Results for the dedicated tab
    concept_tokens_list = []
    for c, tokens in concept_to_tokens.items():
        # Ignore special characters like \u2581 or "_" when counting real tokens
        real_tokens = [t for t in tokens if t not in ["\u2581", "_"]]
        concept_tokens_list.append(
            {"concept": c, "tokens": tokens, "count": len(real_tokens)}
        )
    concept_tokens_list.sort(key=lambda x: x["count"], reverse=True)

    # Build the collapsed Multi-Token Concepts section for Findings
    multi_token_html = ""
    multi_tokens = [ct for ct in concept_tokens_list if ct["count"] > 1]
    if multi_tokens:
        rows = "".join(
            f"<tr><td style='padding: 8px; border-bottom: 1px solid #30363d;'>{ct['concept']}</td>"
            f"<td style='padding: 8px; border-bottom: 1px solid #30363d;'>{ct['count']}</td>"
            f"<td style='padding: 8px; border-bottom: 1px solid #30363d; font-family: monospace;'>{str(ct['tokens'])}</td></tr>"
            for ct in multi_tokens
        )
        multi_token_html = f"""
                <details style="margin-bottom: 40px; background: #0d1117; border: 1px solid #30363d; border-radius: 8px; padding: 10px;">
                    <summary style="cursor: pointer; padding: 10px; font-weight: bold; font-size: 1.1rem; color: #c9d1d9; outline: none;">Multi-Token Concepts (T5 Tokenizer)</summary>
                    <div style="padding: 10px;">
                        <p style="color: #8b949e; margin-top: 0;">These concepts are split into multiple text tokens by the T5 encoder, which may result in incomplete Concept Attention when only the first token is mapped.</p>
                        <table style="width: 100%; border-collapse: collapse; text-align: left;">
                            <thead>
                                <tr style="background: #161b22;">
                                    <th style="padding: 10px; border-bottom: 2px solid #30363d;">Concept</th>
                                    <th style="padding: 10px; border-bottom: 2px solid #30363d;">Token Count</th>
                                    <th style="padding: 10px; border-bottom: 2px solid #30363d;">Token Splits</th>
                                </tr>
                            </thead>
                            <tbody>
                                {rows}
                            </tbody>
                        </table>
                    </div>
                </details>
        """

    html_content += f"""
                {multi_token_html}
                <h2 style="margin-top:0;">Key Findings & Highlights</h2>
                <p style="color: #8b949e; margin-bottom: 30px;">This section highlights specific cases with commentary, or randomly sampled examples from the dataset.</p>
                <div class="findings-grid">
    """

    # Generate Findings Content
    highlighted_keys = [
        "blue_cat",
        "dog_human_face",
        "violence_fans",
        "purple_banana",
        "violence_war",
    ]

    def render_heatmaps(set_path, meta):
        block = '<div class="heatmap-grid">'
        sam_analysis_dir = set_path / "sam_analysis"
        sam_metrics_path = sam_analysis_dir / "metrics.json"

        all_sam_metrics = {}
        if sam_metrics_path.exists():
            try:
                with open(sam_metrics_path, "r") as f:
                    all_sam_metrics = json.load(f)
            except Exception:
                pass

        concepts = meta.get("concepts", [])
        for concept in concepts:
            tokens = concept_to_tokens.get(concept, [])
            tokens_html = f'<div style="color: #8b949e; font-size: 0.8rem; font-family: monospace; margin-top: 2px;">Tokens: {json.dumps(tokens)}</div>'
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

            # Default to masked image if it exists, otherwise upscaled, otherwise original
            primary_display = (
                masked_path
                if masked_path.exists()
                else (upscaled_path if upscaled_path.exists() else orig_heatmap_path)
            )

            sam_data = all_sam_metrics.get(concept)

            block += f"""
                <div class="concept-block">
                    <div class="concept-label">{concept}</div>
                    {tokens_html}
                    <div class="viz-grid">
                        <div class="viz-item">
                            <img src="{get_image_src(primary_display)}" alt="{concept}" loading="lazy">
                            <div class="viz-label">Primary View</div>
                        </div>
                        <details class="viz-item">
                            <summary>View Alternate Heatmaps</summary>
                            <div class="heatmap-toggles">
                                <div>
                                    <img src="{get_image_src(upscaled_path)}" alt="Upscaled {concept}" loading="lazy">
                                    <div class="viz-label">Raw Attention (1024x1024)</div>
                                </div>
                            </div>
                        </details>
                    </div>
            """

            if sam_data:
                iou = sam_data.get("iou", 0)
                precision = sam_data.get("precision", 0)
                recall = sam_data.get("recall", 0)

                sam_viz = (
                    f'<img src="{get_image_src(sam_masked_path)}" alt="SAM Match {concept}" style="max-width: 100px; border-radius: 4px; border: 1px solid #30363d;" loading="lazy">'
                    if sam_masked_path.exists()
                    else ""
                )

                block += f"""
                    <div class="sam-results">
                        <div class="sam-header">
                            <span>SAM Comparison (#{",".join(map(str, sam_data.get("segment_indices", []))) if "segment_indices" in sam_data else sam_data.get("segment_index")})</span>
                            <span>IoU: <span class="metric-val {get_metric_class(iou)}">{iou:.3f}</span></span>
                        </div>
                        <div style="display: flex; gap: 10px; padding: 10px; background: #0d1117; align-items: center; border-bottom: 1px solid #30363d;">
                            {sam_viz}
                            <div class="sam-metrics" style="flex: 1; padding: 0;">
                            <div class="metric-item">
                                <div class="metric-label">IoU</div>
                                <div class="metric-val {get_metric_class(iou)}">{iou:.2%}</div>
                            </div>
                            <div class="metric-item">
                                <div class="metric-label">Precision</div>
                                <div class="metric-val {get_metric_class(precision)}">{precision:.2%}</div>
                            </div>
                            <div class="metric-item">
                                <div class="metric-label">Recall</div>
                                <div class="metric-val {get_metric_class(recall)}">{recall:.2%}</div>
                            </div>
                        </div>
                    </div>
                </div>
                """
            block += "</div>"
        block += "</div>"
        return block

    def render_aggregated_case_block(sets_for_case):
        """Render a single case with the prompt/image at the top, and all sets nested inside"""
        if not sets_for_case:
            return ""

        first_set_str, first_meta = sets_for_case[0]
        first_set_path = Path(first_set_str)
        seed = first_set_path.parent.name
        case = first_set_path.parent.parent.name
        image_path = first_set_path / "image.png"
        coverage_img_path = first_set_path / "sam_analysis" / "debug_coverage_gaps.png"

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

        # Collect all comments from sets
        comments = [m.get("comment", "") for _, m in sets_for_case if m.get("comment")]
        comments_html = ""
        if comments:
            comments_html = f"""
                <div class="comment-section">
                    <span class="comment-label">💡 Observation</span>
                    <div class="comment-area">{"<br>".join(comments)}</div>
                </div>
            """

        # Collect all concepts and categories across all sets for searching/filtering
        all_concepts_for_search = set()
        all_categories = set()
        for _, m in sets_for_case:
            concepts = m.get("concepts", [])
            all_concepts_for_search.update(concepts)
            for c in concepts:
                all_categories.add(get_concept_class(c))

        concepts_search_str = " ".join(all_concepts_for_search).lower()
        categories_str = " ".join(all_categories).lower()

        block = f"""
            <div class="case-block" data-search-text="{prompt.lower()} {case.lower()} {concepts_search_str}" data-categories="{categories_str}">
                <div class="case-info" style="display: flex; justify-content: space-between; align-items: start;">
                    <div>
                        <div class="case-title">{case} | {seed}</div>
                        <div class="prompt">"{prompt}"</div>
                    </div>
                    <button onclick="viewInGrids('{case.lower()}')" style="background: #21262d; border: 1px solid #30363d; color: #58a6ff; padding: 6px 12px; border-radius: 4px; cursor: pointer; font-weight: bold; transition: background 0.2s;">View Grid →</button>
                </div>
                
                <div class="image-grid" style="display: flex; gap: 20px;">
                    <div class="main-image-container" style="text-align: left; margin-bottom: 20px;">
                        <img src="{get_image_src(image_path)}" alt="Generated Image" style="max-width: 400px; border-radius: 8px;" loading="lazy">
                        <div style="color: #8b949e; font-size: 0.85rem; margin-top: 8px;">Generated Image</div>
                    </div>
                    <div class="main-image-container" style="text-align: left; margin-bottom: 20px;">
                        <img src="{get_image_src(coverage_img_path)}" alt="SAM Coverage" style="max-width: 400px; border-radius: 8px;" loading="lazy">
                        <div style="color: #8b949e; font-size: 0.85rem; margin-top: 8px;">SAM Coverage ({coverage_percent:.1f}%)</div>
                    </div>
                </div>
                
                <details class="sets-details" open>
                    <summary class="sets-summary">Concept Sets ({len(sets_for_case)})</summary>
                    <div class="sets-container">
        """

        for set_path_str, meta in sets_for_case:
            set_path = Path(set_path_str)
            set_idx = set_path.name
            concepts = meta.get("concepts", [])
            concepts_str = ", ".join(f"'{c}'" for c in concepts)

            block += f"""
                        <div class="set-block">
                            <div class="set-title">{set_idx}: [{concepts_str}]</div>
                            {render_heatmaps(set_path, meta)}
                        </div>
            """

        block += f"""
                    </div>
                </details>
                {comments_html}
            </div>
        """
        return block

    findings_grids_dir = Path("experiments/results_heatmap_grids")
    if findings_grids_dir.exists():
        findings_grid_files = sorted(findings_grids_dir.glob("*.png"))
        for grid_file in findings_grid_files:
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

                html_content += f"""
                    <div class="case-block" data-search-text="{filename.lower()}">
                        <div class="case-info" style="display: flex; justify-content: space-between; align-items: start;">
                            <div class="case-title">Highlight: {search_term}</div>
                            <button onclick="viewInGallery('{search_term.lower()}')" style="background: #21262d; border: 1px solid #30363d; color: #58a6ff; padding: 6px 12px; border-radius: 4px; cursor: pointer; font-weight: bold; transition: background 0.2s;">View in Gallery →</button>
                        </div>
                        <div style="text-align: center; background: #161b22; padding: 15px; border: 1px solid #30363d; border-radius: 8px;">
                            <img src="{get_image_src(grid_file)}" alt="{filename}" style="max-width: 100%; height: auto; border-radius: 4px;" loading="lazy">
                        </div>
                    </div>
                """

    html_content += """
                </div>
            </div>
            
            <!-- GALLERY TAB -->
            <div id="gallery" class="tab-content">
                <div class="search-container">
                    <input type="text" id="search-input" class="search-input" placeholder="Search concepts, prompts, or case names...">
                    <div class="category-filters">
                        <span style="color: #8b949e; align-self: center; font-size: 0.85rem; margin-right: 5px;">Filter by category:</span>
                        <button class="cat-btn" data-category="object">Object</button>
                        <button class="cat-btn" data-category="color">Color</button>
                        <button class="cat-btn" data-category="texture">Texture</button>
                        <button class="cat-btn" data-category="abstract">Abstract</button>
                        <button class="cat-btn" data-category="spatial">Spatial</button>
                    </div>
                </div>
                <h2 style="margin-top:0;">Complete Gallery</h2>
    """

    # Generate Gallery Content grouped by section
    for idx, group_name in enumerate(
        tqdm(sorted_group_names, desc="Generating Gallery Sections")
    ):
        set_paths_in_group = sorted(groups[group_name])

        html_content += f"""
            <div class="group-section" id="group-{idx}">
                <h2>{group_name}</h2>
        """

        # Group sets by case (prompt/seed) to aggregate them
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

        for case_id, sets_for_case in cases_in_group.items():
            html_content += render_aggregated_case_block(sets_for_case)

        html_content += "</div>"

    html_content += """
            </div>
            
            <!-- GRIDS TAB -->
            <div id="grids" class="tab-content">
                <div class="search-container">
                    <input type="text" id="grid-search-input" class="search-input" placeholder="Search grid images...">
                </div>
                <h2 style="margin-top:0;">Heatmap Grids</h2>
                <div class="findings-grid" id="grids-container">
    """

    grids_dir = Path("experiments/results_heatmap_grids")
    if grids_dir.exists():
        grid_files = sorted(grids_dir.glob("*.png"))
        for grid_file in grid_files:
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

            html_content += f"""
                    <div class="case-block" data-search-text="{filename.lower()}">
                        <div class="case-info" style="display: flex; justify-content: space-between; align-items: start;">
                            <div class="case-title">{filename}</div>
                            <button onclick="viewInGallery('{search_term.lower()}')" style="background: #21262d; border: 1px solid #30363d; color: #58a6ff; padding: 6px 12px; border-radius: 4px; cursor: pointer; font-weight: bold; transition: background 0.2s;">View in Gallery →</button>
                        </div>
                        <div style="text-align: center; background: #161b22; padding: 15px; border: 1px solid #30363d; border-radius: 8px;">
                            <img src="{get_image_src(grid_file)}" alt="{filename}" style="max-width: 100%; height: auto; border-radius: 4px;" loading="lazy">
                        </div>
                    </div>
            """
    else:
        html_content += '<p style="color: #8b949e;">No grid images found in results_heatmap_grids directory.</p>'

    html_content += """
            </div>
            </div>
            
            <!-- TOKENIZATION TAB -->
            <div id="tokenization" class="tab-content">
                <h2 style="margin-top:0;">T5 Tokenization Glossary</h2>
                <p style="color: #8b949e; margin-bottom: 30px;">This table lists all unique concepts found across the generated dataset, showcasing how the T5 encoder splits them into text tokens. Ordered by token count descending.</p>
                <div style="background: #0d1117; border: 1px solid #30363d; border-radius: 8px; padding: 20px;">
                    <table style="width: 100%; border-collapse: collapse; text-align: left;">
                        <thead>
                            <tr style="background: #161b22;">
                                <th style="padding: 10px; border-bottom: 2px solid #30363d;">Concept</th>
                                <th style="padding: 10px; border-bottom: 2px solid #30363d;">Token Count</th>
                                <th style="padding: 10px; border-bottom: 2px solid #30363d;">Token Splits</th>
                            </tr>
                        </thead>
                        <tbody>
"""
    if concept_tokens_list:
        for ct in concept_tokens_list:
            html_content += f"<tr><td style='padding: 8px; border-bottom: 1px solid #30363d;'>{ct['concept']}</td><td style='padding: 8px; border-bottom: 1px solid #30363d;'>{ct['count']}</td><td style='padding: 8px; border-bottom: 1px solid #30363d; font-family: monospace;'>{str(ct['tokens'])}</td></tr>"
    else:
        html_content += "<tr><td colspan='3' style='padding: 8px;'>No concepts found or tokenizer failed to load.</td></tr>"

    html_content += """
                        </tbody>
                    </table>
                </div>
            </div>
            
        </div>
    </div>
    
    <button id="scrollToTopBtn" style="display: none; position: fixed; bottom: 30px; right: 30px; background: #58a6ff; color: #0d1117; border: none; border-radius: 50%; width: 50px; height: 50px; cursor: pointer; font-size: 1.5rem; font-weight: bold; box-shadow: 0 4px 12px rgba(0,0,0,0.5); z-index: 1000; align-items: center; justify-content: center;">↑</button>
"""

    html_content += (
        """
    <script>
        const contentArea = document.querySelector('.content-area');
        const scrollBtn = document.getElementById('scrollToTopBtn');
        
        contentArea.addEventListener('scroll', () => {
            if (contentArea.scrollTop > 300) {
                scrollBtn.style.display = 'flex';
            } else {
                scrollBtn.style.display = 'none';
            }
        });
        
        scrollBtn.addEventListener('click', () => {
            contentArea.scrollTo({top: 0, behavior: 'smooth'});
        });

        // Setup Charts
        function createCharts(classData, metricsCanvasId, countsCanvasId, titlePrefix) {
            const labels = Object.keys(classData);
            const iouData = labels.map(l => classData[l].iou);
            const precData = labels.map(l => classData[l].precision);
            const recData = labels.map(l => classData[l].recall);
            const countData = labels.map(l => classData[l].count);
            
            const customTooltip = {
                callbacks: {
                    label: function(context) {
                        let label = context.dataset.label || '';
                        let cClass = labels[context.dataIndex];
                        let val = context.parsed.y;
                        
                        let dataObj = classData[cClass];
                        let minVal = 0, maxVal = 0, stdVal = 0;
                        if (label === 'IoU') {
                            minVal = dataObj.iou_min; maxVal = dataObj.iou_max; stdVal = dataObj.iou_std;
                        } else if (label === 'Precision') {
                            minVal = dataObj.prec_min; maxVal = dataObj.prec_max; stdVal = dataObj.prec_std;
                        } else if (label === 'Recall') {
                            minVal = dataObj.rec_min; maxVal = dataObj.rec_max; stdVal = dataObj.rec_std;
                        }
                        
                        return [
                            `${label}: ${val.toFixed(3)}`,
                            `  Std: \u00b1${stdVal.toFixed(3)}`,
                            `  Min: ${minVal.toFixed(3)}, Max: ${maxVal.toFixed(3)}`
                        ];
                    }
                }
            };

            new Chart(document.getElementById(metricsCanvasId), {
                type: 'bar',
                data: {
                    labels: labels.map(l => l.charAt(0).toUpperCase() + l.slice(1)),
                    datasets: [
                        { label: 'IoU', data: iouData, backgroundColor: '#58a6ff' },
                        { label: 'Precision', data: precData, backgroundColor: '#3fb950' },
                        { label: 'Recall', data: recData, backgroundColor: '#d29922' }
                    ]
                },
                options: {
                    responsive: true,
                    plugins: {
                        title: { display: true, text: titlePrefix + ' - Metrics (Hover for Details)', color: '#c9d1d9' },
                        legend: { labels: { color: '#8b949e' } },
                        tooltip: customTooltip
                    },
                    scales: {
                        y: { beginAtZero: true, max: 1, ticks: { color: '#8b949e' }, grid: { color: '#30363d' } },
                        x: { ticks: { color: '#8b949e' }, grid: { color: '#30363d' } }
                    }
                }
            });

            new Chart(document.getElementById(countsCanvasId), {
                type: 'pie',
                data: {
                    labels: labels.map(l => l.charAt(0).toUpperCase() + l.slice(1)),
                    datasets: [{
                        data: countData,
                        backgroundColor: ['#58a6ff', '#3fb950', '#d29922', '#f85149', '#a371f7'],
                        borderColor: '#161b22',
                        borderWidth: 2
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        title: { display: true, text: titlePrefix + ' - Count of Concepts', color: '#c9d1d9' },
                        legend: { position: 'right', labels: { color: '#8b949e' } }
                    }
                }
            });
        }

        try {
            const singleData = """
        + json.dumps(single_token_averages)
        + """;
            const multiData = """
        + json.dumps(multi_token_averages)
        + """;
            
            createCharts(singleData, 'singleMetricsChart', 'singleCountsChart', 'Single-Token');
            createCharts(multiData, 'multiMetricsChart', 'multiCountsChart', 'Multi-Token');
        } catch (e) {
            console.error('Error rendering charts:', e);
        }

        // UMAP Rendering
        let umapChartInstance = null;
        try {
            const umapDataRaw = """
        + json.dumps(umap_js_data)
        + """;
            if (umapDataRaw && umapDataRaw.length > 0) {
                const colors = {
                    'object': '#58a6ff',
                    'color': '#3fb950',
                    'texture': '#d29922',
                    'abstract': '#f85149',
                    'spatial': '#a371f7'
                };
                
                function getIoUColor(iou) {
                    const r = Math.round(255 * (1 - iou));
                    const g = Math.round(255 * iou);
                    return `rgb(${r}, ${g}, 50)`;
                }
                
                const datasets = Object.keys(colors).map(cat => {
                    const points = umapDataRaw.filter(p => p.category === cat);
                    return {
                        label: cat.charAt(0).toUpperCase() + cat.slice(1),
                        data: points.map(p => ({ x: p.x, y: p.y, label: p.label, iou: p.iou, category: p.category })),
                        backgroundColor: colors[cat],
                        borderColor: colors[cat],
                        pointRadius: 6,
                        pointHoverRadius: 9
                    };
                });
                
                umapChartInstance = new Chart(document.getElementById('umapChart'), {
                    type: 'scatter',
                    data: { datasets: datasets },
                    options: {
                        responsive: true,
                        maintainAspectRatio: false,
                        plugins: {
                            legend: { position: 'right', labels: { color: '#8b949e' } },
                            tooltip: {
                                callbacks: {
                                    label: function(context) {
                                        let p = context.raw;
                                        let cat = p.category ? ` | ${p.category.charAt(0).toUpperCase() + p.category.slice(1)}` : '';
                                        let iouText = p.iou !== undefined ? ` | IoU: ${p.iou.toFixed(3)}` : '';
                                        return p.label ? `${p.label}${cat}${iouText}` : `(${p.x.toFixed(2)}, ${p.y.toFixed(2)})${cat}${iouText}`;
                                    }
                                }
                            }
                        },
                        scales: {
                            x: { ticks: { color: '#8b949e' }, grid: { color: '#30363d' } },
                            y: { ticks: { color: '#8b949e' }, grid: { color: '#30363d' } }
                        }
                    }
                });
                
                document.getElementById('umapColorToggle').addEventListener('change', (e) => {
                    const mode = e.target.value;
                    if (!umapChartInstance) return;
                    
                    umapChartInstance.data.datasets.forEach(dataset => {
                        if (mode === 'category') {
                            const catColor = colors[dataset.label.toLowerCase()];
                            dataset.backgroundColor = catColor;
                            dataset.borderColor = catColor;
                        } else if (mode === 'iou') {
                            dataset.backgroundColor = dataset.data.map(p => getIoUColor(p.iou));
                            dataset.borderColor = dataset.data.map(p => getIoUColor(p.iou));
                        }
                    });
                    umapChartInstance.update();
                });
            }
        } catch(e) {
            console.error('Error rendering UMAP charts:', e);
        }
        function viewInGallery(searchTerm) {
            const galleryBtn = Array.from(document.querySelectorAll('.tab-btn')).find(b => b.textContent.includes('Gallery'));
            if(galleryBtn) galleryBtn.click();
            
            // Set search query and trigger filter
            const searchInput = document.getElementById('search-input');
            searchInput.value = searchTerm;
            searchInput.dispatchEvent(new Event('input', { bubbles: true }));
            
            // Scroll to top
            contentArea.scrollTo({top: 0, behavior: 'smooth'});
        }

        function viewInGrids(searchTerm) {
            // Find and click the grids tab button
            const gridsBtn = Array.from(document.querySelectorAll('.tab-btn')).find(b => b.textContent.includes('Grids'));
            if(gridsBtn) gridsBtn.click();
            
            const searchInput = document.getElementById('grid-search-input');
            
            // Need to transform internal gallery case names (e.g. 'bicycle_geometry') into potential Grid filenames.
            // Grids might just share part of the name. We'll strip seed and setup a loose query string.
            let query = searchTerm.trim().toLowerCase();
            
            searchInput.value = query;
            searchInput.dispatchEvent(new Event('input', { bubbles: true }));
            
            contentArea.scrollTo({top: 0, behavior: 'smooth'});
        }

        function switchTab(tabId) {
            // Update buttons
            document.querySelectorAll('.tab-btn').forEach(btn => btn.classList.remove('active'));
            event.target.classList.add('active');
            
            // Update content visibility
            document.querySelectorAll('.tab-content').forEach(content => content.classList.remove('active'));
            document.getElementById(tabId).classList.add('active');
            
            // Toggle sidebar visibility only for gallery
            document.getElementById('gallery-sidebar').style.display = (tabId === 'gallery') ? 'block' : 'none';
        }
        
        // Search and Category filtering for Gallery
        const searchInput = document.getElementById('search-input');
        const sections = document.querySelectorAll('.group-section');
        const categoryButtons = document.querySelectorAll('.cat-btn');
        let activeCategories = new Set();

        function updateGalleryFilter() {
            const query = searchInput.value.toLowerCase().trim();
            const searchWords = query.split(/\s+/).filter(word => word.length > 0);
            
            sections.forEach(section => {
                let sectionHasVisibleCase = false;
                const blocks = section.querySelectorAll('.case-block');
                
                blocks.forEach(block => {
                    const searchText = block.getAttribute('data-search-text') || "";
                    const blockCategories = (block.getAttribute('data-categories') || "").split(' ');
                    
                    // Category Filter (OR logic within categories)
                    let catMatch = activeCategories.size === 0;
                    if (!catMatch) {
                        for (let cat of activeCategories) {
                            if (blockCategories.includes(cat)) {
                                catMatch = true;
                                break;
                            }
                        }
                    }

                    // Text Search (AND logic across words)
                    let textMatch = true;
                    if (searchWords.length > 0) {
                        for (let word of searchWords) {
                            if (!searchText.includes(word)) {
                                textMatch = false;
                                break;
                            }
                        }
                    }
                    
                    if (catMatch && textMatch) {
                        block.style.display = 'block';
                        sectionHasVisibleCase = true;
                    } else {
                        block.style.display = 'none';
                    }
                });
                
                if (sectionHasVisibleCase || (query === "" && activeCategories.size === 0)) {
                    section.style.display = 'block';
                } else {
                    section.style.display = 'none';
                }
            });
        }

        searchInput.addEventListener('input', updateGalleryFilter);

        categoryButtons.forEach(btn => {
            btn.addEventListener('click', () => {
                const cat = btn.getAttribute('data-category');
                if (activeCategories.has(cat)) {
                    activeCategories.delete(cat);
                    btn.classList.remove('active');
                } else {
                    activeCategories.add(cat);
                    btn.classList.add('active');
                }
                updateGalleryFilter();
            });
        });
        
        // Search functionality for Grids
        const gridSearchInput = document.getElementById('grid-search-input');
        const gridBlocks = document.querySelectorAll('#grids-container .case-block');

        gridSearchInput.addEventListener('input', (e) => {
            const query = e.target.value.toLowerCase().trim();
            const searchWords = query.split(/\s+/).filter(word => word.length > 0);
            
            gridBlocks.forEach(block => {
                const searchText = block.getAttribute('data-search-text') || "";
                
                let match = true;
                if (searchWords.length > 0) {
                    for (let word of searchWords) {
                        if (!searchText.includes(word)) {
                            match = false;
                            break;
                        }
                    }
                }
                
                if (match) {
                    block.style.display = 'block';
                } else {
                    block.style.display = 'none';
                }
            });
        });
    </script>
</body>
</html>
"""
    )

    with open(output_html, "w", encoding="utf-8") as f:
        f.write(html_content)
    print(f"Gallery generated at {output_html}")


if __name__ == "__main__":
    results_dir = "results/object_analysis"
    rel_html = os.path.join(results_dir, "evaluation_gallery.html")
    generate_gallery(results_dir, rel_html, mode="relative")

    full_html = os.path.join(results_dir, "evaluation_gallery_full.html")
    generate_gallery(results_dir, full_html, mode="base64")
