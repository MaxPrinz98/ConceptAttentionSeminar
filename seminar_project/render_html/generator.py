"""
Main gallery generator — orchestrates all modules.

This is the single entry-point for producing the evaluation HTML.
It calls into data_loading, tokenization, metrics, image_utils,
html_templates, and html_sections to assemble the final document.
"""

import json
from pathlib import Path

from render_html.data_loading import (
    discover_cases,
    collect_unique_concepts,
    load_umap_data,
)
from render_html.tokenization import (
    load_tokenizer,
    compute_concept_tokens,
    build_concept_tokens_list,
)
from render_html.metrics import (
    compute_averages,
    collect_metrics_and_candidates,
)
from render_html.image_utils import make_image_src_resolver
from render_html.html_templates import (
    get_html_header_nav,
    get_gallery_navigator,
    get_explanation_block,
    get_discussion_tab_html,
    get_implementation_details_html,
    get_metrics_tab_html,
    get_statistics_section_html,
    get_umap_section_html,
    get_failure_tab_header,
    get_javascript,
)
from render_html.html_sections import (
    render_top_candidates,
    render_failure_cards,
    render_findings_grid_highlights,
    render_gallery_sections,
    render_grids_section,
    render_multi_token_section,
    render_tokenization_table,
    render_explained_case_block,
)


def generate_gallery(results_dir, output_html, mode="relative"):
    """
    Generate the full evaluation HTML gallery.

    Parameters
    ----------
    results_dir : str or Path
        Root directory containing the analysis results.
    output_html : str or Path
        Path where the HTML file will be written.
    mode : str
        Image source mode — ``"relative"``, ``"absolute"``, or ``"base64"``.
    """
    results_dir = Path(results_dir)
    print(
        f"Generating hierarchical gallery with findings at {output_html} (mode={mode})..."
    )

    # --- Data Loading ---
    groups, all_cases = discover_cases(results_dir)
    unique_concepts = collect_unique_concepts(all_cases)

    # --- Tokenization ---
    t5_tokenizer = load_tokenizer()
    concept_to_tokens = compute_concept_tokens(t5_tokenizer, unique_concepts)

    # --- Metrics ---
    (
        single_token_metrics,
        multi_token_metrics,
        concept_iou_aggregates,
        top_iou_candidates,
        hallucination_candidates,
        missed_localization_candidates,
    ) = collect_metrics_and_candidates(all_cases, concept_to_tokens)

    single_token_averages = compute_averages(single_token_metrics)
    multi_token_averages = compute_averages(multi_token_metrics)

    # --- UMAP ---
    umap_js_data = load_umap_data(results_dir, concept_iou_aggregates)

    # --- Image resolver ---
    get_image_src = make_image_src_resolver(mode, output_html)

    # --- Tokenization list ---
    concept_tokens_list = build_concept_tokens_list(concept_to_tokens)

    # --- Sort groups & render gallery first (to get ToC data) ---
    sorted_group_names = sorted(groups.keys())
    gallery_html, toc_data = render_gallery_sections(
        groups, sorted_group_names, get_image_src, concept_to_tokens
    )

    # =====================================================================
    # Assemble HTML
    # =====================================================================
    html_content = get_html_header_nav("Concept Attention Analysis")
    html_content += '\n    <div class="main-container">'
    html_content += '\n        <div class="content-area">'

    # --- OVERVIEW TAB (formerly Findings) ---
    html_content += "\n            <!-- OVERVIEW TAB -->"
    html_content += '\n            <div id="overview" class="tab-content active">'
    html_content += "\n                <!-- NEW EXPLANATION BLOCK -->"
    html_content += get_explanation_block()
    html_content += get_statistics_section_html()
    html_content += get_umap_section_html()

    # Top candidates
    top_iou_candidates.sort(key=lambda x: x["iou"], reverse=True)
    top_prec_candidates = sorted(
        top_iou_candidates, key=lambda x: x["precision"], reverse=True
    )
    top_rec_candidates = sorted(
        top_iou_candidates, key=lambda x: x["recall"], reverse=True
    )

    html_content += render_top_candidates(
        top_iou_candidates,
        "Intersection over Union (IoU)",
        "iou",
        "#58a6ff",
        get_image_src,
        concept_to_tokens,
    )
    html_content += render_top_candidates(
        top_prec_candidates,
        "Precision",
        "precision",
        "#3fb950",
        get_image_src,
        concept_to_tokens,
    )
    html_content += render_top_candidates(
        top_rec_candidates,
        "Recall",
        "recall",
        "#d29922",
        get_image_src,
        concept_to_tokens,
    )

    # Multi-token collapsed section
    multi_token_html = render_multi_token_section(concept_tokens_list)
    html_content += f"""
                {multi_token_html}
                <h2 style="margin-top:0;">Evaluation Highlights</h2>
                <p style="color: #8b949e; margin-bottom: 30px;">This section highlights specific cases and insights derived from the evaluation.</p>
                <div class="findings-grid">
    """

    # We need to construct the 5 specific cases from all_cases based on their names and evaluated concepts
    # all_cases is a list of tuples: (case_path_str, metadata_dict)

    cases_to_find = [
        (
            "blue_cat_yellow_sofa",
            ["animal", "blue", "sofa", "yellow"],
            "<div style='margin-bottom: 15px; padding: 15px; background: rgba(88, 166, 255, 0.1); border-left: 4px solid #58a6ff; border-radius: 4px;'><p style='margin: 0; color: #c9d1d9;'><b>Observation:</b> Here, the concept of the sofa takes away almost all the attention from the concept <i>yellow</i>. We hypothesize that this relates to the self-attention mechanism of the concept tokens (where concept self-attentions help repel different concepts).</p></div>",
        ),
        (
            "multicolor_blocks",
            ["red", "green", "blue", "white"],
            "<div style='margin-bottom: 15px; padding: 15px; background: rgba(63, 185, 80, 0.1); border-left: 4px solid #3fb950; border-radius: 4px;'><p style='margin: 0; color: #c9d1d9;'><b>Observation:</b> In this example, the distinct primary colors (red, green, blue, white) are separated exceptionally well by the concept attention mechanism.</p></div>",
        ),
        (
            "multicolor_blocks",
            ["top", "middle", "bottom", "white"],
            "<div style='margin-bottom: 15px; padding: 15px; background: rgba(248, 81, 73, 0.1); border-left: 4px solid #f85149; border-radius: 4px;'><p style='margin: 0; color: #c9d1d9;'><b>Observation:</b> You can see that spatial concepts alone do not yield good results. Since we only evaluate single-token concepts currently, we cannot search for the holistic concept \"top block\". The spatial attention alone fails to make up for this context loss.<br><br><b>Follow-up Note:</b> Notice how object concepts (like \"white\") seem to dominate the attention maps if they are included alongside ambiguous spatial terms.</p></div>",
        ),
        (
            "multicolor_blocks",
            ["module", "background", "red", "green", "blue", "white"],
            '<div style=\'margin-bottom: 15px; padding: 15px; background: rgba(248, 81, 73, 0.1); border-left: 4px solid #f85149; border-radius: 4px;\'><p style=\'margin: 0; color: #c9d1d9;\'><b>Observation:</b> You cannot actually search for concepts that occupy the same image regions. For example, the concept "module" takes almost all the attention and leaves very little for the specific colors "red", "green", and "blue".</p></div>',
        ),
        (
            "purple_banana_yellow_grapes",
            ["purple", "banana", "yellow", "grapes"],
            "<div style='margin-bottom: 15px; padding: 15px; background: rgba(210, 153, 34, 0.1); border-left: 4px solid #d29922; border-radius: 4px;'><p style='margin: 0; color: #c9d1d9;'><b>Observation:</b> Concept attention struggles to differentiate between the \"purple banana\" and \"yellow grapes\" representations in the attention maps, even though the diffusion model correctly generated the finalized image conforming to the prompt.</p></div>",
        ),
        (
            "transparent_blue_bird",
            ["glass", "wings", "white", "branch"],
            "<div style='margin-bottom: 15px; padding: 15px; background: rgba(163, 113, 247, 0.1); border-left: 4px solid #a371f7; border-radius: 4px;'><p style='margin: 0; color: #c9d1d9;'><b>Observation:</b> This serves as a strong positive example where the concept heatmaps cleanly and accurately dissect the composition.</p></div>",
        ),
    ]

    for case_name, concepts_req, explanation in cases_to_find:
        # Sort concepts to compare against meta["concepts"] easily
        sorted_req = sorted(concepts_req)
        matching_sets = []
        for path_str in all_cases:
            p = Path(path_str)
            # parent.parent.name is the case name (e.g. blue_cat_yellow_sofa)
            if p.parent.parent.name == case_name:
                meta_path = p / "metadata.json"
                if meta_path.exists():
                    try:
                        import json as _json

                        with open(meta_path, "r") as f:
                            meta = _json.load(f)
                    except Exception:
                        continue

                    meta_concepts = meta.get("concepts", [])
                    if sorted(meta_concepts) == sorted_req:
                        matching_sets.append((path_str, meta))

        if matching_sets:
            # We found matching sets for this case
            case_slug = f"highlight-{case_name}-{'_'.join(sorted_req)}"
            html_content += render_explained_case_block(
                case_slug, matching_sets, explanation, get_image_src, concept_to_tokens
            )

    html_content += (
        """
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
                    <div style="display: flex; gap: 20px; align-items: center; margin-top: 12px; flex-wrap: wrap;">
                        <div style="display: flex; align-items: center; gap: 10px;">
                            <label style="color: #8b949e; font-size: 0.85rem; white-space: nowrap;">Jump to:</label>
                            """
        + get_gallery_navigator(toc_data)
        + """
                        </div>
                        <div class="size-slider-container" style="margin-top: 0;">
                            <label>Image size:</label>
                            <input type="range" id="img-size-slider" class="size-slider" min="120" max="500" value="250" step="10">
                            <span id="img-size-val" class="size-slider-val">250px</span>
                        </div>
                    </div>
                </div>
                <h2 style="margin-top:0;">Complete Gallery</h2>
    """
    )

    # Gallery sections (already rendered above)
    html_content += gallery_html

    # --- FAILURE ANALYSIS TAB ---
    hallucination_candidates.sort(key=lambda x: x["iou"])
    missed_localization_candidates.sort(key=lambda x: x["iou"])

    failure_halluc_html = render_failure_cards(
        hallucination_candidates, "#f85149", get_image_src
    )
    failure_missed_html = render_failure_cards(
        missed_localization_candidates, "#d29922", get_image_src
    )

    html_content += """
            </div>
            """
    html_content += get_failure_tab_header(
        len(hallucination_candidates), len(missed_localization_candidates)
    )
    html_content += failure_halluc_html
    html_content += f"""
                
                <h3 style="color: #d29922; margin-top: 40px; border-bottom: 1px solid #d2992233; padding-bottom: 10px;">Missed Localizations (showing worst {min(10, len(missed_localization_candidates))} of {len(missed_localization_candidates)})</h3>
                {failure_missed_html}
            </div>
    """

    # --- GRIDS TAB ---
    grids_dir = results_dir.parent / "results_heatmap_grids"
    html_content += """
            <!-- GRIDS TAB -->
            <div id="grids" class="tab-content">
                <div class="search-container">
                    <input type="text" id="grid-search-input" class="search-input" placeholder="Search grid images...">
                </div>
                <h2 style="margin-top:0;">Heatmap Grids</h2>
                <div class="findings-grid" id="grids-container">
    """
    html_content += render_grids_section(grids_dir, get_image_src)
    html_content += """
            </div>
            </div>
    """

    # --- METRICS TAB ---
    html_content += get_metrics_tab_html(get_image_src)

    # --- DISCUSSION TAB ---
    html_content += get_discussion_tab_html()

    # --- IMPLEMENTATION TAB ---
    html_content += get_implementation_details_html()

    # --- TOKENIZATION TAB ---
    html_content += """
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
    html_content += render_tokenization_table(concept_tokens_list)
    html_content += """
                        </tbody>
                    </table>
                </div>
            </div>
            
        </div>
    </div>
    
    <button id="scrollToTopBtn" style="display: none; position: fixed; bottom: 30px; right: 30px; background: #58a6ff; color: #0d1117; border: none; border-radius: 50%; width: 50px; height: 50px; cursor: pointer; font-size: 1.5rem; font-weight: bold; box-shadow: 0 4px 12px rgba(0,0,0,0.5); z-index: 1000; align-items: center; justify-content: center;">↑</button>
"""

    # JavaScript
    html_content += get_javascript(
        single_token_averages, multi_token_averages, umap_js_data
    )

    with open(output_html, "w", encoding="utf-8") as f:
        f.write(html_content)
    print(f"Gallery generated at {output_html}")
