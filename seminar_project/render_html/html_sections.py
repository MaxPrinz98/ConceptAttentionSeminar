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

        # Fallback if masked image not present
        if not masked_img.exists():
            masked_img = case_path / f"upscaled_heatmap_{cand['safe_concept']}.png"
            if not masked_img.exists():
                masked_img = case_path / f"heatmap_{cand['safe_concept']}.png"

        tokens = concept_to_tokens.get(cand["concept"], [])
        tokens_json = json.dumps(tokens)

        case_name = Path(cand["case_path"]).parent.parent.name

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
                            <div style="display: flex; flex-direction: column; gap: 8px; align-items: flex-end;">
                                <div style="display: flex; gap: 10px;">
                                    <div class="metric-item" title="IoU"><span class="metric-label">IoU</span><br><span class="metric-val {get_metric_class(cand["iou"])}">{cand["iou"]:.2%}</span></div>
                                    <div class="metric-item" title="Precision"><span class="metric-label">Prec</span><br><span class="metric-val {get_metric_class(cand["precision"])}">{cand["precision"]:.2%}</span></div>
                                    <div class="metric-item" title="Recall"><span class="metric-label">Rec</span><br><span class="metric-val {get_metric_class(cand["recall"])}">{cand["recall"]:.2%}</span></div>
                                </div>
                                <div style="display: flex; gap: 6px;">
                                    <button onclick="viewInGallery('{case_name.lower()}')" style="background: #21262d; border: 1px solid #30363d; color: #58a6ff; padding: 4px 10px; border-radius: 4px; cursor: pointer; font-size: 0.8rem; transition: background 0.2s;">Gallery →</button>
                                    <button onclick="viewInGrids('{case_name.lower()}')" style="background: #21262d; border: 1px solid #30363d; color: #58a6ff; padding: 4px 10px; border-radius: 4px; cursor: pointer; font-size: 0.8rem; transition: background 0.2s;">Grid →</button>
                                </div>
                            </div>
                        </div>
                        
                        <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 15px;">
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
    """Render the heatmap grid: original image + one column per concept, side by side."""
    sam_analysis_dir = set_path / "sam_analysis"
    sam_metrics_path = sam_analysis_dir / "metrics.json"

    all_sam_metrics = {}
    if sam_metrics_path.exists():
        try:
            with open(sam_metrics_path, "r") as f:
                all_sam_metrics = json.load(f)
        except Exception:
            pass

    # Original image as first column
    image_path = set_path / "image.png"
    
    # Check for Argmax Segmentation Heatmap
    seg_heatmap_path = set_path / "heatmap_segmentation.png"
    seg_legend_path = set_path / "segmentation_legend.json"
    
    seg_html = ""
    if seg_heatmap_path.exists() and seg_legend_path.exists():
        try:
            with open(seg_legend_path, "r") as f:
                legend_data = json.load(f)
                
            legend_items_html = ""
            for name, color in legend_data.items():
                legend_items_html += f"""
                    <div style="display: flex; align-items: center; gap: 6px; margin-bottom: 4px;">
                        <span style="display: inline-block; width: 12px; height: 12px; background-color: {color}; border: 1px solid #30363d;"></span>
                        <span style="color: #c9d1d9; font-size: 0.8rem;">{name}</span>
                    </div>
                """
                
            seg_html = f"""
                <div class="viz-item" style="margin-top: 20px; padding-top: 15px; border-top: 1px dashed #30363d;">
                    <img src="{get_image_src(seg_heatmap_path)}" alt="Argmax Segmentation" loading="lazy" style="border: 1px solid #30363d;">
                    <div class="viz-label" style="color: #c9d1d9;">Attention Segmentation</div>
                    <div style="margin-top: 10px; text-align: left; padding: 0 10px;">
                        {legend_items_html}
                    </div>
                </div>
            """
        except Exception:
            pass
            
    block = '<div class="heatmap-grid">'
    block += f"""
                <div class="concept-block" style="min-width: 250px;">
                    <div class="concept-label">Original</div>
                    <div class="viz-item">
                        <img src="{get_image_src(image_path)}" alt="Generated Image" loading="lazy">
                        <div class="viz-label">Generated Image</div>
                    </div>
                    {seg_html}
                </div>
    """

    concepts = meta.get("concepts", [])
    for concept in concepts:
        tokens = concept_to_tokens.get(concept, [])
        tokens_html = f'<div style="color: #8b949e; font-size: 0.75rem; font-family: monospace; margin-top: 2px; text-align: center;">Tokens: {json.dumps(tokens)}</div>'
        safe_c = safe_concept_name(concept)
        masked_path = set_path / f"masked_image_{safe_c}.png"
        upscaled_path = set_path / f"upscaled_heatmap_{safe_c}.png"
        orig_heatmap_path = set_path / f"heatmap_{safe_c}.png"
        sam_masked_path = sam_analysis_dir / f"matched_masked_image_{safe_c}.png"

        upscaled_display = upscaled_path if upscaled_path.exists() else orig_heatmap_path
        masked_display = masked_path if masked_path.exists() else upscaled_display

        sam_data = all_sam_metrics.get(concept)

        block += f"""
                <div class="concept-block">
                    <div class="concept-label">{concept}</div>
                    {tokens_html}
                    <div class="viz-item">
                        <img src="{get_image_src(upscaled_display)}" alt="Heatmap {concept}" loading="lazy">
                        <div class="viz-label">Upscaled Heatmap</div>
                    </div>
                    <div class="viz-item">
                        <img src="{get_image_src(masked_display)}" alt="Masked {concept}" loading="lazy">
                        <div class="viz-label">Masked Image</div>
                    </div>
        """

        if sam_data:
            iou = sam_data.get("iou", 0)
            precision = sam_data.get("precision", 0)
            recall = sam_data.get("recall", 0)
            segment_indices = sam_data.get("segment_indices", [])
            segments_str = ", ".join(map(str, segment_indices)) if segment_indices else "None"
            
            diagnostic_path = sam_analysis_dir / f"overlap_diagnostic_{safe_c}.png"

            block += f"""
                    <div class="viz-item">
                        <img src="{get_image_src(sam_masked_path)}" alt="SAM {concept}" loading="lazy">
                        <div class="viz-label">Composite SAM Segment</div>
                        <div style="color: #8b949e; font-size: 0.75rem; text-align: center; margin-top: 2px;">Ids: [{segments_str}]</div>
                    </div>
                    <div style="display: flex; gap: 4px; justify-content: center; margin-bottom: 8px;">
                        <div class="metric-item" style="flex:1;"><div class="metric-label">IoU</div><div class="metric-val {get_metric_class(iou)}">{iou:.2%}</div></div>
                        <div class="metric-item" style="flex:1;"><div class="metric-label">Prec</div><div class="metric-val {get_metric_class(precision)}">{precision:.2%}</div></div>
                        <div class="metric-item" style="flex:1;"><div class="metric-label">Rec</div><div class="metric-val {get_metric_class(recall)}">{recall:.2%}</div></div>
                    </div>
            """
            
            if diagnostic_path.exists():
                block += f"""
                    <div class="viz-item" style="margin-top: 5px;">
                        <img src="{get_image_src(diagnostic_path)}" alt="Diagnostic {concept}" loading="lazy" style="border: 2px solid #58a6ff;">
                        <div class="viz-label" style="color: #c9d1d9;">Composite Diagnostic</div>
                    </div>
                """
                
            # SINGLE BEST SEGMENT BLOCK
            single_data = sam_data.get("single_best_segment")
            if single_data:
                single_iou = single_data.get("iou", 0)
                single_prec = single_data.get("precision", 0)
                single_rec = single_data.get("recall", 0)
                single_idx = single_data.get("index", "Unknown")
                
                single_masked_path = sam_analysis_dir / f"single_matched_masked_image_{safe_c}.png"
                single_diag_path = sam_analysis_dir / f"single_overlap_diagnostic_{safe_c}.png"
                
                if single_masked_path.exists() and single_diag_path.exists():
                    block += f"""
                    <div style="margin-top: 20px; padding-top: 15px; border-top: 1px dashed #30363d;">
                        <div class="viz-item">
                            <img src="{get_image_src(single_masked_path)}" alt="Single SAM {concept}" loading="lazy">
                            <div class="viz-label" style="color: #8b949e;">Best Single Segment</div>
                            <div style="color: #8b949e; font-size: 0.75rem; text-align: center; margin-top: 2px;">Id: [{single_idx}]</div>
                        </div>
                        <div style="display: flex; gap: 4px; justify-content: center; margin-bottom: 8px;">
                            <div class="metric-item" style="flex:1; padding: 4px;"><div class="metric-label" style="font-size:0.7rem;">IoU</div><div class="metric-val {get_metric_class(single_iou)}" style="font-size:0.9rem;">{single_iou:.2%}</div></div>
                            <div class="metric-item" style="flex:1; padding: 4px;"><div class="metric-label" style="font-size:0.7rem;">Prec</div><div class="metric-val {get_metric_class(single_prec)}" style="font-size:0.9rem;">{single_prec:.2%}</div></div>
                            <div class="metric-item" style="flex:1; padding: 4px;"><div class="metric-label" style="font-size:0.7rem;">Rec</div><div class="metric-val {get_metric_class(single_rec)}" style="font-size:0.9rem;">{single_rec:.2%}</div></div>
                        </div>
                        <div class="viz-item" style="margin-top: 5px;">
                            <img src="{get_image_src(single_diag_path)}" alt="Single Diagnostic {concept}" loading="lazy" style="border: 1px solid #7a5ea6;">
                            <div class="viz-label" style="color: #8b949e; font-size: 0.8rem;">Single Diagnostic</div>
                        </div>
                    </div>
                    """

        block += "</div>"
    block += "</div>"
    return block


# ---------------------------------------------------------------------------
# Aggregated Case Block (used in Gallery and Findings)
# ---------------------------------------------------------------------------

def render_aggregated_case_block(case_id_slug, sets_for_case, get_image_src, concept_to_tokens):
    """Render a single case with the prompt at the top and tabbed concept sets."""
    if not sets_for_case:
        return ""

    first_set_str, first_meta = sets_for_case[0]
    first_set_path = Path(first_set_str)
    seed = first_set_path.parent.name
    case = first_set_path.parent.parent.name

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
            <div class="case-block" id="{case_id_slug}" data-search-text="{prompt.lower()} {case.lower()} {concepts_search_str}" data-categories="{categories_str}">
                <div class="case-info" style="display: flex; justify-content: space-between; align-items: start;">
                    <div>
                        <div class="case-title">{case} | {seed}</div>
                        <div class="prompt">"{prompt}"</div>
                    </div>
                    <button onclick="viewInGrids('{case.lower()}')" style="background: #21262d; border: 1px solid #30363d; color: #58a6ff; padding: 6px 12px; border-radius: 4px; cursor: pointer; font-weight: bold; transition: background 0.2s;">View Grid →</button>
                </div>
    """

    # Build concept-set view
    if len(sets_for_case) == 1:
        # Single set — no tabs needed
        set_path_str, meta = sets_for_case[0]
        set_path = Path(set_path_str)
        concepts = meta.get("concepts", [])
        concepts_str = ", ".join(f"'{c}'" for c in concepts)
        block += f"""
                <div style="margin-top: 10px;">
                    <div class="set-title">[{concepts_str}]</div>
                    {render_heatmaps(set_path, meta, get_image_src, concept_to_tokens)}
                </div>
        """
    else:
        # Multiple sets — render as tabs
        tab_buttons = ""
        tab_panels = ""
        for i, (set_path_str, meta) in enumerate(sets_for_case):
            set_path = Path(set_path_str)
            concepts = meta.get("concepts", [])
            concepts_str = ", ".join(f"'{c}'" for c in concepts)
            active_cls = " active" if i == 0 else ""
            tab_buttons += f'<button class="concept-tab-btn{active_cls}" onclick="switchConceptTab(\'{case_id_slug}\', {i})">[{concepts_str}]</button>'
            tab_panels += f"""
                <div class="concept-tab-panel{active_cls}">
                    {render_heatmaps(set_path, meta, get_image_src, concept_to_tokens)}
                </div>
            """

        block += f"""
                <div class="concept-tabs" id="{case_id_slug}-tabs">
                    {tab_buttons}
                </div>
                {tab_panels}
        """

    block += f"""
                {comments_html}
            </div>
    """
    return block


def render_explained_case_block(case_id_slug, sets_for_case, explanation_html, get_image_src, concept_to_tokens):
    """Render a single case with the prompt at the top and tabbed concept sets, alongside a specific user explanation."""
    base_block = render_aggregated_case_block(case_id_slug, sets_for_case, get_image_src, concept_to_tokens)
    if not base_block:
        return ""
    
    # We inject the explanation HTML at the top of the case-block, right after the opening div
    # The base block starts with: \n            <div class="case-block" id="slug" ...>\n
    # So we split on the first inner div '<div class="case-info"'
    parts = base_block.split('<div class="case-info"', 1)
    if len(parts) == 2:
        return parts[0] + explanation_html + '\n                <div class="case-info"' + parts[1]
    return base_block


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
                    </div>
                </div>
                <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 15px; margin-top: 15px;">
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
    """Render the complete gallery content grouped by section.

    Returns
    -------
    tuple[str, list[dict]]
        The HTML string, and a ``toc_data`` list for the hierarchical sidebar.
    """
    html = ""
    toc_data = []
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

        group_toc = {"group_name": group_name, "group_idx": idx, "cases": []}

        for case_id, sets_for_case in cases_in_group.items():
            case_slug = f"case-{idx}-{case_id}"
            case_display_name = Path(sets_for_case[0][0]).parent.parent.name
            group_toc["cases"].append({"case_name": case_display_name, "case_id": case_slug})
            html += render_aggregated_case_block(
                case_slug, sets_for_case, get_image_src, concept_to_tokens
            )

        toc_data.append(group_toc)
        html += "</div>"

    return html, toc_data


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
