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
    get_css,
    get_html_head,
    get_html_header_nav,
    get_sidebar_toc,
    get_explanation_block,
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
    css = get_css()
    html_content = get_html_head(css)
    html_content += "\n<body>"
    html_content += get_html_header_nav()
    html_content += '\n    <div class="main-container">'
    html_content += "\n        <!-- Sidebar ONLY visible for Gallery -->"
    html_content += get_sidebar_toc(toc_data)
    html_content += '\n        <div class="content-area">'

    # --- FINDINGS TAB ---
    html_content += "\n            <!-- FINDINGS TAB -->"
    html_content += '\n            <div id="findings" class="tab-content active">'
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
                <h2 style="margin-top:0;">Key Findings &amp; Highlights</h2>
                <p style="color: #8b949e; margin-bottom: 30px;">This section highlights specific cases with commentary, or randomly sampled examples from the dataset.</p>
                <div class="findings-grid">
    """

    # Highlighted grids
    highlighted_keys = [
        "blue_cat",
        "dog_human_face",
        "violence_fans",
        "purple_banana",
        "violence_war",
    ]
    findings_grids_dir = Path("results/results_heatmap_grids")
    html_content += render_findings_grid_highlights(
        findings_grids_dir, highlighted_keys, get_image_src
    )

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
                    <div class="size-slider-container">
                        <label>Image size:</label>
                        <input type="range" id="img-size-slider" class="size-slider" min="120" max="500" value="250" step="10">
                        <span id="img-size-val" class="size-slider-val">250px</span>
                    </div>
                </div>
                <h2 style="margin-top:0;">Complete Gallery</h2>
    """

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
    grids_dir = Path("results/results_heatmap_grids")
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
