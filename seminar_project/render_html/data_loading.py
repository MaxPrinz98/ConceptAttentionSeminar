"""
Data loading and directory-walking utilities.

Responsible for discovering experiment cases on disk, reading
metadata files, and loading auxiliary data like UMAP coordinates
and SAM coverage summaries.
"""

import os
import json
from pathlib import Path

from render_html.concept_classes import get_concept_class


def discover_cases(results_dir):
    """
    Walk *results_dir* and collect every ``set_*`` directory.

    Returns
    -------
    groups : dict[str, list[str]]
        Mapping from group name to list of set-directory paths.
    all_cases : list[str]
        Flat list of all set-directory paths.
    """
    results_dir = Path(results_dir)
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
            all_cases.append(root)

    return groups, all_cases


def collect_unique_concepts(all_cases):
    """
    Read metadata from every case and return the union of all concepts.
    """
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
    return unique_concepts


def load_umap_data(results_dir, concept_iou_aggregates):
    """
    Load UMAP coordinates from ``concept_umap.json`` and merge with
    average IoU values.

    Returns a list of dicts suitable for the JavaScript scatter plot.
    """
    umap_path = Path(results_dir) / "concept_umap.json"
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
    return umap_js_data


def load_sam_coverage(case_path):
    """
    Read the SAM coverage percentage from ``segments_summary.json``.

    Returns 0.0 if the file does not exist or cannot be parsed.
    """
    summary_path = Path(case_path) / "sam_analysis" / "segments_summary.json"
    if summary_path.exists():
        try:
            with open(summary_path, "r") as f:
                summary_data = json.load(f)
            return summary_data.get("overall_coverage_percent", 0.0)
        except Exception:
            pass
    return 0.0
