"""
Metric aggregation and candidate collection.

Provides functions for initialising per-class metric buckets,
computing summary statistics (mean, std, min, max), and collecting
top-performing / failure-case candidates from the dataset.
"""

import json
import numpy as np
from pathlib import Path

from render_html.concept_classes import get_concept_class
from render_html.image_utils import safe_concept_name
from render_html.tokenization import filter_real_tokens


def get_empty_metrics():
    """Return a fresh per-class metrics structure with empty lists."""
    return {
        "object": {"iou": [], "precision": [], "recall": []},
        "color": {"iou": [], "precision": [], "recall": []},
        "texture": {"iou": [], "precision": [], "recall": []},
        "abstract": {"iou": [], "precision": [], "recall": []},
        "spatial": {"iou": [], "precision": [], "recall": []},
    }


def compute_averages(metrics_dict):
    """
    Compute mean, std, min, max and count for each concept class.

    Parameters
    ----------
    metrics_dict : dict
        Output of :func:`get_empty_metrics` after being populated.

    Returns
    -------
    dict
        Per-class summary statistics.
    """
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


def collect_metrics_and_candidates(all_cases, concept_to_tokens):
    """
    Iterate over all cases, read SAM metrics, and populate:

    * single / multi-token metric buckets
    * per-concept IoU aggregates
    * top-performing, hallucination, and missed-localisation candidates

    Returns
    -------
    tuple of (single_token_metrics, multi_token_metrics,
              concept_iou_aggregates,
              top_iou_candidates, hallucination_candidates,
              missed_localization_candidates)
    """
    single_token_metrics = get_empty_metrics()
    multi_token_metrics = get_empty_metrics()
    concept_iou_aggregates = {}

    top_iou_candidates = []
    hallucination_candidates = []
    missed_localization_candidates = []

    for case_path_str in all_cases:
        meta_path = Path(case_path_str) / "metadata.json"
        if not meta_path.exists():
            continue

        with open(meta_path, "r") as f:
            meta = json.load(f)

        sam_metrics_path = Path(case_path_str) / "sam_analysis" / "metrics.json"
        if not sam_metrics_path.exists():
            continue

        try:
            with open(sam_metrics_path, "r") as f:
                all_sam_metrics = json.load(f)
        except Exception:
            continue

        for concept in meta.get("concepts", []):
            if concept not in all_sam_metrics:
                continue

            c_class = get_concept_class(concept)
            data = all_sam_metrics[concept]
            iou_val = data.get("iou", 0)
            prec_val = data.get("precision", 0)
            rec_val = data.get("recall", 0)

            tokens = concept_to_tokens.get(concept, [])
            real_tokens = filter_real_tokens(tokens)
            is_multi = len(real_tokens) > 1
            target_metrics = (
                multi_token_metrics if is_multi else single_token_metrics
            )

            target_metrics[c_class]["iou"].append(iou_val)
            target_metrics[c_class]["precision"].append(prec_val)
            target_metrics[c_class]["recall"].append(rec_val)

            if concept not in concept_iou_aggregates:
                concept_iou_aggregates[concept] = []
            concept_iou_aggregates[concept].append(iou_val)

            safe_c = safe_concept_name(concept)

            # Top performances
            if iou_val > 0:
                top_iou_candidates.append(
                    {
                        "case_path": case_path_str,
                        "prompt": meta.get("prompt", ""),
                        "seed": Path(case_path_str).parent.name,
                        "concept": concept,
                        "iou": iou_val,
                        "precision": prec_val,
                        "recall": rec_val,
                        "iou_rank_label": (
                            f"#{','.join(map(str, data.get('segment_indices', [])))} SAM Segs"
                            if "segment_indices" in data
                            else f"#{data.get('segment_index')} SAM Seg"
                        ),
                        "safe_concept": safe_c,
                    }
                )

            # Failure: hallucination (low precision)
            if prec_val < 0.15 and iou_val < 0.1:
                hallucination_candidates.append(
                    {
                        "case_path": case_path_str,
                        "prompt": meta.get("prompt", ""),
                        "seed": Path(case_path_str).parent.name,
                        "concept": concept,
                        "iou": iou_val,
                        "precision": prec_val,
                        "recall": rec_val,
                        "safe_concept": safe_c,
                    }
                )

            # Failure: missed localisation (low recall)
            if rec_val < 0.15 and iou_val < 0.1:
                missed_localization_candidates.append(
                    {
                        "case_path": case_path_str,
                        "prompt": meta.get("prompt", ""),
                        "seed": Path(case_path_str).parent.name,
                        "concept": concept,
                        "iou": iou_val,
                        "precision": prec_val,
                        "recall": rec_val,
                        "safe_concept": safe_c,
                    }
                )

    return (
        single_token_metrics,
        multi_token_metrics,
        concept_iou_aggregates,
        top_iou_candidates,
        hallucination_candidates,
        missed_localization_candidates,
    )
