"""
HTML section rendering functions.

Each function renders a self-contained block of HTML for a specific
part of the evaluation dashboard (top candidates, heatmaps, gallery
cards, failure cards, grids, tokenization table, etc.).
"""

import json
from pathlib import Path

from render_html.concept_classes import get_concept_class
from render_html.image_utils import get_metric_class, safe_concept_name
from render_html.data_loading import load_sam_coverage


# ---------------------------------------------------------------------------
# Top Performing Concepts
# ---------------------------------------------------------------------------

def render_top_candidates(
    candidates, metric_name, metric_key, color_hex,
    get_image_src, concept_to_tokens,
):
    """
    Render a collapsible ``<details>`` block showing the top 15
    candidates ranked by *metric_key*.
    """
    display_candidates = candidates[:15]
    if not display_candidates:
        return ""

    metric_label_map = {"iou": "IoU", "precision": "Prec", "recall": "Rec"}
    m_label = metric_label_map.get(metric_key, metric_key.title())

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

        coverage_percent = load_sam_coverage(case_path)

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
                                <div class="metric-item" title="IoU"><span class="metric-label">IoU</span><br><span class="metric-val {get_metric_class(cand["iou"])}">{cand["iou"]:.2%}</span></div>
                                <div class="metric-item" title="Precision"><span class="metric-label">Prec</span><br><span class="metric-val {get_metric_class(cand["precision"])}">{cand["precision"]:.2%}</span></div>
                                <div class="metric-item" title="Recall"><span class="metric-label">Rec</span><br><span class="metric-val {get_metric_class(cand["recall"])}">{cand["recall"]:.2%}</span></div>
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


# ---------------------------------------------------------------------------
# Concept Heatmaps (inside a case block)
# ---------------------------------------------------------------------------

def render_heatmaps(set_path, meta, get_image_src, concept_to_tokens):
    """Render the heatmap grid for all concepts in a single set."""
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
        safe_c = safe_concept_name(concept)
        masked_path = set_path / f"masked_image_{safe_c}.png"
        upscaled_path = set_path / f"upscaled_heatmap_{safe_c}.png"
        orig_heatmap_path = set_path / f"heatmap_{safe_c}.png"
        sam_masked_path = sam_analysis_dir / f"matched_masked_image_{safe_c}.png"

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


# ---------------------------------------------------------------------------
# Aggregated Case Block (used in Gallery and Findings)
# ---------------------------------------------------------------------------

def render_aggregated_case_block(sets_for_case, get_image_src, concept_to_tokens):
    """Render a single case with the prompt/image at the top and all sets nested inside."""
    if not sets_for_case:
        return ""

    first_set_str, first_meta = sets_for_case[0]
    first_set_path = Path(first_set_str)
    seed = first_set_path.parent.name
    case = first_set_path.parent.parent.name
    image_path = first_set_path / "image.png"
    coverage_img_path = first_set_path / "sam_analysis" / "debug_coverage_gaps.png"

    coverage_percent = load_sam_coverage(first_set_path)

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
                            {render_heatmaps(set_path, meta, get_image_src, concept_to_tokens)}
                        </div>
            """

    block += f"""
                    </div>
                </details>
                {comments_html}
            </div>
    """
    return block


# ---------------------------------------------------------------------------
# Failure Cards
# ---------------------------------------------------------------------------

def render_failure_cards(candidates, color_hex, get_image_src):
    """Render cards for failure-case candidates (hallucinations or missed localisations)."""
    if not candidates:
        return '<p style="color: #8b949e; text-align: center; padding: 20px;">No cases found matching these criteria.</p>'
    cards = ""
    for cand in candidates[:10]:
        cp = Path(cand["case_path"])
        orig_img = cp / "image.png"
        masked_img = cp / f"masked_image_{cand['safe_concept']}.png"
        if not masked_img.exists():
            masked_img = cp / f"upscaled_heatmap_{cand['safe_concept']}.png"
            if not masked_img.exists():
                masked_img = cp / f"heatmap_{cand['safe_concept']}.png"
        sam_img = (
            cp / "sam_analysis" / f"matched_masked_image_{cand['safe_concept']}.png"
        )
        coverage_img = cp / "sam_analysis" / "debug_coverage_gaps.png"

        coverage_percent = load_sam_coverage(cp)

        iou_cls = get_metric_class(cand["iou"])
        prec_cls = get_metric_class(cand["precision"])
        rec_cls = get_metric_class(cand["recall"])

        cards += f'''
            <div class="case-block" style="border-left: 4px solid {color_hex};">
                <div class="case-info" style="display: flex; justify-content: space-between; align-items: center;">
                    <div>
                        <div class="case-title" style="color: {color_hex};">{cand["concept"]}</div>
                        <div class="prompt">"{cand["prompt"]}" | seed_{cand["seed"]}</div>
                    </div>
                    <div style="display: flex; gap: 8px;">
                        <div class="metric-item"><div class="metric-label">IoU</div><div class="metric-val {iou_cls}">{cand["iou"]:.2%}</div></div>
                        <div class="metric-item"><div class="metric-label">Prec</div><div class="metric-val {prec_cls}">{cand["precision"]:.2%}</div></div>
                        <div class="metric-item"><div class="metric-label">Rec</div><div class="metric-val {rec_cls}">{cand["recall"]:.2%}</div></div>
                        <div class="metric-item"><div class="metric-label">Coverage</div><div class="metric-val" style="color: #8b949e;">{coverage_percent:.1f}%</div></div>
                    </div>
                </div>
                <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 15px; margin-top: 15px;">
                    <div style="text-align: center;">
                        <img src="{get_image_src(orig_img)}" style="width: 100%; border-radius: 6px; border: 1px solid #30363d;" loading="lazy">
                        <div style="color: #8b949e; font-size: 0.75rem; margin-top: 5px;">Original</div>
                    </div>
                    <div style="text-align: center;">
                        <img src="{get_image_src(masked_img)}" style="width: 100%; border-radius: 6px; border: 1px solid #30363d;" loading="lazy">
                        <div style="color: #8b949e; font-size: 0.75rem; margin-top: 5px;">Masked Attention</div>
                    </div>
                    <div style="text-align: center;">
                        <img src="{get_image_src(sam_img)}" style="width: 100%; border-radius: 6px; border: 1px solid #30363d;" loading="lazy">
                        <div style="color: #8b949e; font-size: 0.75rem; margin-top: 5px;">SAM Ground Truth</div>
                    </div>
                    <div style="text-align: center;">
                        <img src="{get_image_src(coverage_img)}" style="width: 100%; border-radius: 6px; border: 1px solid #30363d;" loading="lazy">
                        <div style="color: #8b949e; font-size: 0.75rem; margin-top: 5px;">SAM Coverage ({coverage_percent:.1f}%)</div>
                    </div>
                </div>
            </div>
            '''
    return cards


# ---------------------------------------------------------------------------
# Findings Grid Highlights
# ---------------------------------------------------------------------------

def _clean_grid_name(filename):
    """Strip known prefixes and seed suffixes from a grid filename."""
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
            clean_name = clean_name[len(prefix):]
    clean_name = clean_name.split("_seed_")[0]
    return clean_name.replace("_", " ").title()


def render_findings_grid_highlights(findings_grids_dir, highlighted_keys, get_image_src):
    """Render the highlighted grid images in the Findings tab."""
    html = ""
    if not findings_grids_dir.exists():
        return html

    findings_grid_files = sorted(findings_grids_dir.glob("*.png"))
    for grid_file in findings_grid_files:
        filename = grid_file.name
        if any(key in filename for key in highlighted_keys):
            search_term = _clean_grid_name(filename)

            html += f"""
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
    return html


# ---------------------------------------------------------------------------
# Gallery Sections
# ---------------------------------------------------------------------------

def render_gallery_sections(groups, sorted_group_names, get_image_src, concept_to_tokens):
    """Render the complete gallery content grouped by section."""
    html = ""
    from tqdm import tqdm

    for idx, group_name in enumerate(
        tqdm(sorted_group_names, desc="Generating Gallery Sections")
    ):
        set_paths_in_group = sorted(groups[group_name])

        html += f"""
            <div class="group-section" id="group-{idx}">
                <h2>{group_name}</h2>
        """

        # Group sets by case (prompt/seed) to aggregate them
        cases_in_group = {}
        for set_path_str in set_paths_in_group:
            meta_path = Path(set_path_str) / "metadata.json"
            if meta_path.exists():
                with open(meta_path, "r") as f:
                    import json as _json
                    meta = _json.load(f)
                case_id = (
                    Path(set_path_str).parent.parent.name
                    + "_"
                    + Path(set_path_str).parent.name
                )
                if case_id not in cases_in_group:
                    cases_in_group[case_id] = []
                cases_in_group[case_id].append((set_path_str, meta))

        for case_id, sets_for_case in cases_in_group.items():
            html += render_aggregated_case_block(
                sets_for_case, get_image_src, concept_to_tokens
            )

        html += "</div>"
    return html


# ---------------------------------------------------------------------------
# Grids Section
# ---------------------------------------------------------------------------

def render_grids_section(grids_dir, get_image_src):
    """Render the content of the Grids tab."""
    html = ""
    if grids_dir.exists():
        grid_files = sorted(grids_dir.glob("*.png"))
        for grid_file in grid_files:
            filename = grid_file.name
            search_term = _clean_grid_name(filename)

            html += f"""
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
        html += '<p style="color: #8b949e;">No grid images found in results_heatmap_grids directory.</p>'
    return html


# ---------------------------------------------------------------------------
# Multi-Token Concepts (Findings tab collapsible)
# ---------------------------------------------------------------------------

def render_multi_token_section(concept_tokens_list):
    """Render the collapsible multi-token concepts table for the Findings tab."""
    multi_tokens = [ct for ct in concept_tokens_list if ct["count"] > 1]
    if not multi_tokens:
        return ""

    rows = "".join(
        f"<tr><td style='padding: 8px; border-bottom: 1px solid #30363d;'>{ct['concept']}</td>"
        f"<td style='padding: 8px; border-bottom: 1px solid #30363d;'>{ct['count']}</td>"
        f"<td style='padding: 8px; border-bottom: 1px solid #30363d; font-family: monospace;'>{str(ct['tokens'])}</td></tr>"
        for ct in multi_tokens
    )

    return f"""
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


# ---------------------------------------------------------------------------
# Tokenization Tab
# ---------------------------------------------------------------------------

def render_tokenization_table(concept_tokens_list):
    """Render the full tokenization glossary table for the Tokenization tab."""
    html = ""
    if concept_tokens_list:
        for ct in concept_tokens_list:
            html += f"<tr><td style='padding: 8px; border-bottom: 1px solid #30363d;'>{ct['concept']}</td><td style='padding: 8px; border-bottom: 1px solid #30363d;'>{ct['count']}</td><td style='padding: 8px; border-bottom: 1px solid #30363d; font-family: monospace;'>{str(ct['tokens'])}</td></tr>"
    else:
        html += "<tr><td colspan='3' style='padding: 8px;'>No concepts found or tokenizer failed to load.</td></tr>"
    return html
