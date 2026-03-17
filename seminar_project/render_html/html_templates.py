"""
Static HTML/CSS/JS templates.

Contains the CSS stylesheet, HTML skeleton fragments, explanatory
blocks, and the JavaScript code that powers the interactive
dashboard (charts, search, filtering, tab switching).
"""

import json
from pathlib import Path


def get_css():
    """Return the full CSS stylesheet as a string."""
    return """
        :root { --gallery-img-size: 250px; }
        html { scroll-behavior: smooth; }
        body { font-family: 'Inter', system-ui, -apple-system, sans-serif; background: #0f1115; color: #e1e4e8; margin: 0; padding: 0; line-height: 1.6; display: flex; flex-direction: column; height: 100vh; }
        
        .header-nav { background: #161b22; padding: 15px 30px; border-bottom: 1px solid #30363d; display: flex; justify-content: space-between; align-items: center; }
        .header-nav h1 { margin: 0; color: #58a6ff; font-size: 1.5rem; }
        
        .tabs { display: flex; gap: 20px; }
        .tab-btn { background: transparent; border: none; color: #8b949e; font-size: 1.1rem; font-weight: bold; cursor: pointer; padding: 10px 15px; border-bottom: 3px solid transparent; transition: all 0.2s; }
        .tab-btn:hover { color: #c9d1d9; }
        .tab-btn.active { color: #58a6ff; border-bottom-color: #58a6ff; }

        .main-container { display: flex; flex: 1; overflow: hidden; }
        
        /* Gallery navigator dropdown */
        .gallery-nav-select { background: #161b22; border: 1px solid #30363d; color: #c9d1d9; padding: 8px 12px; border-radius: 6px; font-size: 0.85rem; cursor: pointer; min-width: 200px; max-width: 400px; }
        .gallery-nav-select:focus { outline: none; border-color: #58a6ff; }
        
        .content-area { flex: 1; overflow-y: auto; padding: 30px; }
        
        .tab-content { display: none; max-width: 1800px; margin: 0 auto; }
        .tab-content.active { display: block; }

        /* Finding and Gallery Cards */
        .case-block { background: #0d1117; border: 1px solid #30363d; border-radius: 8px; padding: 20px; margin-bottom: 25px; }
        .case-info { margin-bottom: 20px; border-left: 4px solid #58a6ff; padding-left: 15px; }
        .case-title { font-size: 1.2rem; font-weight: bold; color: #c9d1d9; font-family: monospace; }
        .prompt { font-style: italic; color: #8b949e; margin: 8px 0; font-size: 0.95rem; }
        
        .image-grid { display: grid; grid-template-columns: 1fr 2fr; gap: 30px; align-items: start; }
        .main-image-container { text-align: center; }
        .main-image-container img { width: 100%; max-width: 400px; border-radius: 8px; box-shadow: 0 4px 12px rgba(0,0,0,0.5); border: 1px solid #30363d; }
        
        .heatmap-grid { display: flex; gap: 15px; overflow-x: auto; padding-bottom: 10px; }
        .concept-block { background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 15px; display: flex; flex-direction: column; gap: 10px; min-width: var(--gallery-img-size); max-width: calc(var(--gallery-img-size) + 40px); flex: 0 0 auto; }
        .concept-label { font-size: 1.1rem; font-weight: bold; color: #58a6ff; margin-bottom: 5px; border-bottom: 1px solid #30363d; padding-bottom: 5px; text-align: center; }
        
        .viz-grid { display: flex; flex-direction: column; gap: 10px; }
        .viz-item { text-align: center; background: #0d1117; border: 1px solid #30363d; border-radius: 6px; padding: 8px; }
        .viz-item img { width: 100%; height: auto; border-radius: 4px; border: 1px solid #21262d; transition: transform 0.2s; }
        .viz-item img:hover { transform: scale(1.02); border-color: #58a6ff; }

        /* Image size slider */
        .size-slider-container { display: flex; align-items: center; gap: 10px; margin-top: 12px; }
        .size-slider-container label { color: #8b949e; font-size: 0.85rem; white-space: nowrap; }
        .size-slider { -webkit-appearance: none; appearance: none; width: 200px; height: 6px; border-radius: 3px; background: #30363d; outline: none; }
        .size-slider::-webkit-slider-thumb { -webkit-appearance: none; appearance: none; width: 16px; height: 16px; border-radius: 50%; background: #58a6ff; cursor: pointer; }
        .size-slider-val { color: #58a6ff; font-family: monospace; font-size: 0.85rem; min-width: 45px; }
        .viz-label { font-size: 0.8rem; color: #8b949e; margin-top: 5px; }
        
        .sam-results { margin-top: 15px; border: 1px solid #30363d; border-radius: 6px; overflow: hidden; font-size: 0.8rem; }
        .sam-header { background: #161b22; padding: 8px; font-weight: bold; border-bottom: 1px solid #30363d; display: flex; justify-content: space-between; }
        .sam-metrics { display: grid; grid-template-columns: repeat(auto-fit, minmax(100px, 1fr)); gap: 8px; padding: 10px; background: #0d1117; }
        .metric-item { background: #161b22; padding: 8px; border-radius: 4px; border: 1px solid #21262d; text-align: center; }
        .metric-label { font-size: 0.7rem; color: #8b949e; margin-bottom: 2px; text-transform: uppercase; }
        .metric-val { font-weight: bold; font-family: monospace; font-size: 0.9rem; }
        
        .metric-good { color: #3fb950; }
        .metric-fair { color: #d29922; }
        .metric-poor { color: #f85149; }

        .comment-section { margin-top: 25px; padding-top: 20px; border-top: 1px solid #30363d; background: rgba(88, 166, 255, 0.05); padding: 15px; border-radius: 8px; }
        .comment-label { font-weight: bold; color: #58a6ff; margin-bottom: 10px; display: block; font-size: 1rem; }
        .viz-item summary { background: #161b22; border: 1px solid #30363d; border-radius: 4px; padding: 8px; font-size: 0.9rem; color: #c9d1d9; }
        .viz-item summary:hover { background: #30363d; }
        .heatmap-toggles { display: flex; flex-direction: column; gap: 10px; margin-top: 10px; }
        .heatmap-toggles > div { background: #0d1117; border: 1px solid #21262d; border-radius: 4px; padding: 8px; }

        /* Top Search Bar */
        .search-container { padding: 20px 30px; border-bottom: 1px solid #30363d; margin: -30px -30px 30px -30px; background: rgba(15, 17, 21, 0.95); backdrop-filter: blur(8px); position: sticky; top: -30px; z-index: 100; }
        .search-input { width: 100%; max-width: 600px; padding: 12px 20px; border-radius: 8px; border: 1px solid #30363d; background: #161b22; color: #e1e4e8; font-size: 1rem; outline: none; transition: border-color 0.2s; }
        .search-input:focus { border-color: #58a6ff; box-shadow: 0 0 0 3px rgba(88, 166, 255, 0.1); }

        .category-filters { display: flex; gap: 10px; margin-top: 15px; flex-wrap: wrap; }
        .cat-btn { background: #161b22; border: 1px solid #30363d; color: #8b949e; padding: 6px 12px; border-radius: 20px; cursor: pointer; font-size: 0.85rem; transition: all 0.2s; }
        .cat-btn:hover { border-color: #8b949e; color: #c9d1d9; }
        .cat-btn.active { background: #58a6ff33; border-color: #58a6ff; color: #58a6ff; }
        
        .findings-grid { display: grid; grid-template-columns: 1fr; gap: 30px; }
        
        /* Groups for Gallery */
        .group-section { margin-bottom: 50px; padding-top: 20px; }
        .group-section h2 { border-bottom: 2px solid #30363d; padding-bottom: 10px; color: #fff; }
        
        .sets-details { background: transparent; border: 1px solid #30363d; border-radius: 8px; margin-top: 20px; overflow: hidden; }
        .sets-summary { padding: 12px 15px; cursor: pointer; font-weight: 600; font-size: 1rem; background: #21262d; outline: none; display: flex; align-items: center; color: #e1e4e8; }
        .sets-summary:hover { background: #30363d; }
        .sets-summary:after { content: '▸'; margin-left: auto; transition: transform 0.2s; }
        .sets-details[open] .sets-summary:after { transform: rotate(90deg); }
        
        .set-block { border-top: 1px solid #30363d; padding: 20px; background: #0d1117; }
        .set-title { color: #8b949e; font-size: 0.9rem; margin-bottom: 15px; font-weight: bold; text-transform: uppercase; letter-spacing: 1px; }

        /* Concept-set tabs */
        .concept-tabs { display: flex; gap: 0; border-bottom: 2px solid #30363d; margin-top: 20px; overflow-x: auto; }
        .concept-tab-btn { background: transparent; border: none; border-bottom: 3px solid transparent; color: #8b949e; font-size: 0.85rem; font-family: monospace; padding: 10px 16px; cursor: pointer; transition: all 0.2s; white-space: nowrap; }
        .concept-tab-btn:hover { color: #c9d1d9; background: #161b22; }
        .concept-tab-btn.active { color: #58a6ff; border-bottom-color: #58a6ff; background: #0d1117; }
        .concept-tab-panel { display: none; padding: 20px 0; }
        .concept-tab-panel.active { display: block; }


    """




def get_implementation_details_html():
    """Return the HTML for the Implementation Details tab."""
    return """
        <!-- IMPLEMENTATION DETAILS TAB -->
        <div id="implementation" class="tab-content">
            <h2 style="margin-top:0; font-size: 2.5rem;">Pipeline Implementation Details</h2>
            <p style="color: #8b949e; margin-bottom: 20px; font-size: 1.3rem; line-height: 1.6;">This section outlines the core programs in the <code>seminar_project</code> directory, their purpose in the evaluation pipeline, and highlights key algorithmic implementations.</p>
            
            <div style="background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 25px; margin-bottom: 40px;">
                <h3 style="margin-top: 0; color: #c9d1d9; font-size: 1.6rem; border-bottom: 1px solid #30363d; padding-bottom: 10px;">Pipeline Summary: What this project does</h3>
                <ul style="color: #8b949e; font-size: 1.25rem; line-height: 1.7; padding-left: 25px; margin-top: 15px;">
                    <li><strong style="color: #c9d1d9;">Image & Attention Generation:</strong> We hook into the <code>Flux1.schnell</code> transformer during image generation to extract the raw cross-attention probability heatmaps for specific concept tokens.</li>
                    <li><strong style="color: #c9d1d9;">Mask Processing:</strong> We upscale the low-resolution heatmaps and apply adaptive thresholding (Otsu's method) to binarize them into crisp "Concept masks."</li>
                    <li><strong style="color: #c9d1d9;">Argmax Segmentation:</strong> We create a holistic segmentation map by assigning each pixel to the concept that has the highest attention weight at that location.</li>
                    <li><strong style="color: #c9d1d9;">Automated Ground Truth:</strong> We use the Segment Anything Model (SAM) to break the generated images into generic pieces, then apply a greedy algorithm to compose the pieces that best match the concept mask into a "ground truth" shape.</li>
                    <li><strong style="color: #c9d1d9;">Evaluation:</strong> We score the Concept masks against the automated SAM ground truth to calculate Intersection over Union (IoU), Precision, and Recall.</li>
                    <li><strong style="color: #c9d1d9;">Reporting:</strong> We aggregate all generated artifacts, metrics, and text embeddings into this interactive HTML dashboard.</li>
                </ul>
            </div>

            <div style="display: flex; flex-direction: column; gap: 40px;">
                
                <!-- Final Directory Structure Overview -->
                <div class="case-block">
                    <h3 style="margin-top: 0; color: #58a6ff; font-size: 1.8rem;">Final Expected Output Structure</h3>
                    <p style="color: #c9d1d9; line-height: 1.6; font-size: 1.25rem; margin-bottom: 20px;">By the end of the full pipeline execution, each evaluated concept set generates a comprehensive set of artifacts arranged in the following hierarchical structure:</p>
                    <div style="background: #0d1117; padding: 20px; border-radius: 6px; border: 1px solid #30363d; font-family: monospace; font-size: 1.15rem; color: #c9d1d9; overflow-x: auto;">
<pre style="margin: 0; color: #c9d1d9; font-size: 1.15rem;">results/object_analysis/
└── {group_name}/                    <span style="color: #8b949e;"># e.g., attribute_color</span>
    └── {case_name}/                 <span style="color: #8b949e;"># e.g., blue_cat_yellow_sofa</span>
        └── seed_{N}/                <span style="color: #8b949e;"># e.g., seed_0</span>
            └── set_{concepts}/      <span style="color: #8b949e;"># e.g., set_animal_blue_sofa_yellow</span>
                │
                ├── image.png                     <span style="color: #8b949e;"># Original 1024x1024 generated image</span>
                ├── metadata.json                 <span style="color: #8b949e;"># Details about tokens and evaluation params</span>
                │
                ├── heatmap_*.png                 <span style="color: #8b949e;"># Low-res raw attention arrays</span>
                ├── upscaled_heatmap_*.png        <span style="color: #8b949e;"># High-res colorized attention visualization</span>
                ├── mask_*.png                    <span style="color: #8b949e;"># Binarized attention threshold mask</span>
                ├── masked_image_*.png            <span style="color: #8b949e;"># Binary mask overlaid on original image</span>
                ├── heatmap_segmentation.png      <span style="color: #8b949e;"># Argmax categorical segmentation image</span>
                ├── segmentation_legend.json      <span style="color: #8b949e;"># Color mappings for the argmax segmentation</span>
                │
                └── sam_analysis/                 <span style="color: #8b949e;"># Final SAM Evaluation Artifacts</span>
                    ├── metrics.json              <span style="color: #8b949e;"># Contains IoU, Precision, and Recall scores</span>
                    ├── segments_summary.json     <span style="color: #8b949e;"># SAM internal metadata</span>
                    ├── debug_coverage_gaps.png   <span style="color: #8b949e;"># Diagnostic visualization of SAM failures</span>
                    ├── matched_mask_*.png        <span style="color: #8b949e;"># Composed SAM "ground truth" mask</span>
                    ├── matched_masked_image_*.png<span style="color: #8b949e;"># Overlay visualizing the IoU overlap</span>
                    └── all_segments/             <span style="color: #8b949e;"># Raw individual segments parsed by SAM</span></pre>
                    </div>
                </div>

                <!-- Pipeline Overview -->
                <div class="case-block">
                    <h3 style="margin-top: 0; color: #58a6ff; font-size: 1.8rem;">1. pipeline.py</h3>
                    <p style="color: #c9d1d9; line-height: 1.6; font-size: 1.25rem;">The central orchestrator of the entire evaluation process. It runs the scripts in a sequential cascade to ensure all data is generated, processed, evaluated, and finally rendered into this HTML dashboard. It allows running specific steps or resuming from failures.</p>
                </div>

                <!-- Run Object Analysis -->
                <div class="case-block">
                    <h3 style="margin-top: 0; color: #58a6ff; font-size: 1.8rem;">2. run_object_analysis.py</h3>
                    <p style="color: #c9d1d9; line-height: 1.6; font-size: 1.25rem; margin-bottom: 20px;">This script handles the image generation and raw concept attention extraction. It loads the <code>Flux1.schnell</code> model, parses the <code>experiments.json</code> configurations, and hooks into the transformer's double stream attention blocks (layers 16-18) to extract the cross-attention probabilities for specific concept tokens.</p>
                    <div style="background: #0d1117; padding: 20px; border-radius: 6px; border: 1px solid #30363d; font-family: monospace; font-size: 1.15rem; color: #c9d1d9;">
                        <span style="color: #8b949e; display: block; margin-bottom: 15px; font-size: 1.25rem; text-transform: uppercase;">Generated File Tree</span>
<pre style="margin: 0; color: #c9d1d9; font-size: 1.15rem;">results/object_analysis/
└── {group_name}/                    <span style="color: #8b949e;"># e.g., attribute_color</span>
    └── {case_name}/                 <span style="color: #8b949e;"># e.g., blue_cat_yellow_sofa</span>
        └── seed_{N}/                <span style="color: #8b949e;"># e.g., seed_0</span>
            └── set_{concepts}/      <span style="color: #8b949e;"># e.g., set_animal_blue_sofa_yellow</span>
                ├── image.png        <span style="color: #8b949e;"># The 1024x1024 generated image</span>
                ├── metadata.json    <span style="color: #8b949e;"># Run parameters & tokenizer mapping</span>
                └── heatmap_*.png    <span style="color: #8b949e;"># Raw low-res attention arrays</span></pre>
                    </div>
                </div>

                <!-- Process Heatmaps -->
                <div class="case-block">
                    <h3 style="margin-top: 0; color: #58a6ff; font-size: 1.8rem;">3. process_heatmaps.py</h3>
                    <p style="color: #c9d1d9; line-height: 1.6; font-size: 1.25rem;">Converts the raw, low-resolution attention heatmaps into usable formats. It upscales them to 1024x1024, applies the 'inferno' colormap, and generates a binary mask used for quantitative evaluation.</p>
                    
                    <div style="background: #0d1117; padding: 20px; border-radius: 6px; border: 1px solid #30363d; margin-top: 20px; overflow-x: auto;">
                        <span style="color: #8b949e; font-size: 1.25rem; display: block; margin-bottom: 12px;">Dynamic Mask Thresholding snippet:</span>
<pre style="margin: 0; color: #c9d1d9; font-family: monospace; font-size: 1.15rem;"><code># 1. Normalize heatmap
heatmap_np = np.array(upscaled_heatmap).astype(float) / 255.0

# 2. Dynamic thresholding: Otsu's method on the non-zero attention values
nonzero_vals = heatmap_np[heatmap_np > 0.05]
if len(nonzero_vals) > 0:
    threshold = threshold_otsu(nonzero_vals)
else:
    threshold = 0.5  # Fallback

# 3. Create binary mask
mask = heatmap_np > threshold
mask_img = Image.fromarray((mask * 255).astype(np.uint8))</code></pre>
                        <div style="margin-top: 20px; padding-top: 20px; border-top: 1px solid #30363d;">
                            <p style="color: #8b949e; font-size: 1.2rem; line-height: 1.6; margin: 0;"><strong>How it works:</strong><br>
                            Unlike a static threshold (like fixing it at 0.5), this code adapts to each heatmap. <br>
                            • Line 5 ignores all background pixels with &lt;5% attention. If we included the massive background of zeros, the thresholding algorithm would skew heavily and fail.<br>
                            • Line 7 uses <em>Otsu's Method</em>, an algorithm that analyzes the variance in pixel intensities to mathematically find the optimal dividing line between "foreground object" and "background".<br>
                            • Line 12 essentially says "Any pixel brighter than Otsu's threshold is part of the concept mask (True), everything else is background (False)".
                            </p>
                        </div>
                    </div>

                    <div style="background: #0d1117; padding: 20px; border-radius: 6px; border: 1px solid #30363d; margin-top: 20px; overflow-x: auto;">
                        <span style="color: #8b949e; font-size: 1.25rem; display: block; margin-bottom: 12px;">Argmax Segmentation logic:</span>
<pre style="margin: 0; color: #c9d1d9; font-family: monospace; font-size: 1.15rem;"><code># 1. Stack all individual concept heatmaps and add a base background layer
bg_layer = np.full(img_size[::-1], 0.1)  # 10% attention threshold for background
stacked = np.stack([bg_layer] + heatmaps, axis=0)

# 2. Determine the dominant concept per pixel
segmentation_idx = np.argmax(stacked, axis=0)

# 3. Apply distinct categorical colors
cmap = cm.get_cmap("tab10")
colors = [(0, 0, 0)]  # Background is black
for i in range(len(concept_names)):
    r, g, b, _ = cmap((i % 10) / 10.0)
    colors.append((int(r * 255), int(g * 255), int(b * 255)))

seg_rgb = np.zeros((*img_size[::-1], 3), dtype=np.uint8)
for i, color in enumerate(colors):
    seg_rgb[segmentation_idx == i] = color

Image.fromarray(seg_rgb).save("heatmap_segmentation.png")</code></pre>
                        <div style="margin-top: 20px; padding-top: 20px; border-top: 1px solid #30363d;">
                            <p style="color: #8b949e; font-size: 1.2rem; line-height: 1.6; margin: 0;"><strong>How it works:</strong><br>
                            To get a holistic view of how the model separates visual features, we combine all attention heatmaps into a single image.<br>
                            • We build a 3D stack of all the concepts plus a generic background threshold.<br>
                            • `np.argmax` scans the "Z" axis for every pixel and returns the index of the concept that has the strongest weight. If no concept surpasses the 0.1 threshold, the pixel is assigned to Background.<br>
                            • The categorical mapping applies discrete 'tab10' colors mapped into a <code>segmentation_legend.json</code> reference file.
                            </p>
                        </div>
                    </div>

                    <div style="background: #0d1117; padding: 20px; border-radius: 6px; border: 1px solid #30363d; margin-top: 20px; font-family: monospace; font-size: 1.15rem; color: #c9d1d9;">
                        <span style="color: #8b949e; display: block; margin-bottom: 15px; font-size: 1.25rem; text-transform: uppercase;">Generated File Tree (Appended)</span>
<pre style="margin: 0; color: #c9d1d9; font-size: 1.15rem;">results/object_analysis/
└── {group_name}/
    └── {case_name}/
        └── seed_{N}/
            └── set_{concepts}/
                ├── mask_*.png                 <span style="color: #8b949e;"># Binary threshold mask</span>
                ├── masked_image_*.png         <span style="color: #8b949e;"># Mask overlaid on original image</span>
                ├── upscaled_heatmap_*.png     <span style="color: #8b949e;"># Colorized attention visualization</span>
                ├── heatmap_segmentation.png   <span style="color: #8b949e;"># Argmax categorical segmentation image</span>
                └── segmentation_legend.json   <span style="color: #8b949e;"># Color mappings for the argmax segmentation</span></pre>
                    </div>
                </div>

                <!-- Run SAM Analysis -->
                <div class="case-block">
                    <h3 style="margin-top: 0; color: #58a6ff; font-size: 1.8rem;">4. run_sam_analysis.py</h3>
                    <p style="color: #c9d1d9; line-height: 1.6; font-size: 1.25rem; margin-bottom: 20px;">Evaluates the concept attention masks against an automated "ground truth" generated by the Segment Anything Model (SAM). Since SAM doesn't know <em>what</em> an object is, we use a greedy algorithm to compose the optimal combination of generic SAM segments that align with the Concept Attention mask.</p>
                    
                    <div style="background: #0d1117; padding: 20px; border-radius: 6px; border: 1px solid #30363d; margin-top: 20px; overflow-x: auto;">
                        <span style="color: #8b949e; font-size: 1.25rem; display: block; margin-bottom: 12px;">Greedy SAM Segment Composition snippet:</span>
<pre style="margin: 0; color: #c9d1d9; font-family: monospace; font-size: 1.15rem;"><code>current_composite_mask = np.zeros_like(concept_mask)
selected_indices = []
best_iou = 0.0

while True:
    improved = False
    best_temp_iou = best_iou
    best_temp_idx = -1

    # 1. Try adding each unused SAM segment
    for i in range(num_segments):
        if i in selected_indices: continue
        
        temp_mask = np.logical_or(current_composite_mask, sam_segment_masks[i])
        temp_iou = calculate_iou(concept_mask, temp_mask)

        if temp_iou > best_temp_iou:
            best_temp_iou = temp_iou
            best_temp_idx = i

    # 2. If IoU improved, permanently add the segment to the composite
    if best_temp_idx != -1:
        current_composite_mask = np.logical_or(
            current_composite_mask, sam_segment_masks[best_temp_idx]
        )
        selected_indices.append(best_temp_idx)
        best_iou = best_temp_iou
        improved = True

    # 3. Stop if no segment improved the IoU
    if not improved:
        break</code></pre>
                        <div style="margin-top: 20px; padding-top: 20px; border-top: 1px solid #30363d;">
                            <p style="color: #8b949e; font-size: 1.2rem; line-height: 1.6; margin: 0;"><strong>How it works:</strong><br>
                            SAM breaks the image into many tiny generic pieces (e.g., a car might be split into wheels, doors, windows). The concept attention mask highlights the whole car, but isn't a perfect bounding shape.<br>
                            • The `while` loop continuously tries to build a better "ground truth" shape.<br>
                            • Block 1 tests <em>every single remaining SAM piece</em> by overlaying it onto our running composite mask, and calculates the new Intersection-over-Union (IoU) overlap score with the Concept mask.<br>
                            • Block 2 locks in the single SAM piece that provided the biggest boost to the overall IoU score. The very first piece chosen is also recorded as the <strong>"Single Best Segment"</strong> to diagnose how a single standalone SAM piece compares to the merged composite mask.<br>
                            • Block 3 breaks the loop when adding another piece actually hurts the IoU score (meaning the piece doesn't belong to the concept we're looking for).
                            </p>
                        </div>
                    </div>

                    <div style="background: #0d1117; padding: 20px; border-radius: 6px; border: 1px solid #30363d; margin-top: 20px; font-family: monospace; font-size: 1.15rem; color: #c9d1d9;">
                        <span style="color: #8b949e; display: block; margin-bottom: 15px; font-size: 1.25rem; text-transform: uppercase;">Generated File Tree (Appended)</span>
<pre style="margin: 0; color: #c9d1d9; font-size: 1.15rem;">results/object_analysis/
└── {group_name}/
    └── {case_name}/
        └── seed_{N}/
            └── set_{concepts}/
                └── sam_analysis/
                    ├── all_segments/                 <span style="color: #8b949e;"># Raw SAM segments (segment_X and masked_segment_X)</span>
                    ├── debug_coverage_gaps.png       <span style="color: #8b949e;"># Visualizes areas missed by SAM</span>
                    ├── matched_mask_*.png            <span style="color: #8b949e;"># Composed SAM mask matching concept</span>
                    ├── matched_masked_image_*.png    <span style="color: #8b949e;"># Green/Red visual of IoU</span>
                    ├── metrics.json                  <span style="color: #8b949e;"># Crucial numerical scores (IoU, Precision, Recall)</span>
                    └── segments_summary.json         <span style="color: #8b949e;"># Metadata about the extracted segments</span></pre>
                    </div>
                </div>

                <!-- Evaluate / Render HTML -->
                <div class="case-block">
                    <h3 style="margin-top: 0; color: #58a6ff; font-size: 1.8rem;">5. evaluate_object_analysis_html.py & render_html/</h3>
                    <p style="color: #c9d1d9; line-height: 1.6; font-size: 1.25rem;">The final step dynamically crawls the entire <code>results/</code> directory structure, reading every <code>metadata.json</code> and `sam_analysis/metrics.json` file it finds. It computes aggregate statistics (averages, top performers, failures) across all experimental runs, handles tokenization introspection via the T5 encoder, and finally injects all the gathered data into a comprehensive interactive HTML dashboard (like this one).</p>
                </div>

                <!-- Extract Embeddings (Bonus) -->
                <div class="case-block">
                    <h3 style="margin-top: 0; color: #58a6ff; font-size: 1.8rem;">6. extract_embeddings.py</h3>
                    <p style="color: #c9d1d9; line-height: 1.6; font-size: 1.25rem;">An auxiliary script handling the semantic clustering visualization in the Findings tab. It extracts the raw 4096-dimensional text embeddings from the T5 encoder for every concept, projects them down to 2 dimensions using UMAP (Uniform Manifold Approximation and Projection), and saves a <code>results_umap/concept_embeddings.json</code> file which is ingested by the dashboard's Chart.js visualization engine.</p>
                </div>

            </div>
        </div>
    """


def get_metrics_tab_html(get_image_src):
    """Return the HTML for the Metrics explanation tab."""
    base_path = Path("results/object_analysis/attribute_color/blue_cat_yellow_sofa/seed_0/set_animal_blue_sofa_yellow")
    # If paths don't exist (e.g., partial runs), fallback to empty placeholders, but we assume they exist for the main report.
    concept_mask_src = get_image_src(base_path / "masked_image_animal.png") if (base_path / "masked_image_animal.png").exists() else ""
    sam_mask_src = get_image_src(base_path / "sam_analysis" / "matched_masked_image_animal.png") if (base_path / "sam_analysis" / "matched_masked_image_animal.png").exists() else ""
    overlap_vis_src = get_image_src(base_path / "sam_analysis" / "overlap_diagnostic_animal.png") if (base_path / "sam_analysis" / "overlap_diagnostic_animal.png").exists() else ""

    return f"""
        <!-- METRICS TAB -->
        <div id="metrics" class="tab-content">
            <h2 style="margin-top:0; font-size: 2.5rem;">Understanding the Evaluation Metrics</h2>
            <p style="color: #8b949e; margin-bottom: 30px; font-size: 1.3rem; line-height: 1.6;">To evaluate the concept attention mathematically, we compare the thresholded attention mask (our "Prediction") against a composite Segment Anything (SAM) mask composed of generic segments (our "Ground Truth").</p>
            
            <div style="display: flex; gap: 40px; flex-wrap: wrap;">
                <!-- Left Column: Metrics Definitions -->
                <div style="flex: 1; min-width: 400px; display: flex; flex-direction: column; gap: 30px;">
                    <div class="case-block" style="border-left: 4px solid #58a6ff;">
                        <h3 style="margin-top: 0; color: #58a6ff; font-size: 1.8rem;">Intersection over Union (IoU)</h3>
                        <p style="color: #c9d1d9; line-height: 1.6; font-size: 1.25rem;">The primary metric of success. It measures the overall overlap between the Concept Mask and the SAM Ground Truth.<br>
                        <strong>Score in example:</strong> 0.92</p>
                        <div style="background: #0d1117; padding: 15px; border-radius: 6px; font-family: monospace; font-size: 1.15rem; color: #8b949e;">
                            IoU = (Area of Overlap) / (Area of Union)
                        </div>
                    </div>

                    <div class="case-block" style="border-left: 4px solid #3fb950;">
                        <h3 style="margin-top: 0; color: #3fb950; font-size: 1.8rem;">Precision</h3>
                        <p style="color: #c9d1d9; line-height: 1.6; font-size: 1.25rem;">Measures how "clean" the attention mask is. Of all the pixels the Concept Mask highlighted, what percentage actually belong to the ground truth object? A low score means the attention "leaked" into the background (False Positives).<br>
                        <strong>Score in example:</strong> 0.98</p>
                        <div style="background: #0d1117; padding: 15px; border-radius: 6px; font-family: monospace; font-size: 1.15rem; color: #8b949e;">
                            Precision = (Area of Overlap) / (Area of Concept Mask)
                        </div>
                    </div>

                    <div class="case-block" style="border-left: 4px solid #d29922;">
                        <h3 style="margin-top: 0; color: #d29922; font-size: 1.8rem;">Recall (Coverage)</h3>
                        <p style="color: #c9d1d9; line-height: 1.6; font-size: 1.25rem;">Measures how completely the attention mask covers the object. Of all the pixels in the ground truth object, what percentage did the Concept Mask successfully find? A low score means the attention missed parts of the object (False Negatives).<br>
                        <strong>Score in example:</strong> 0.94</p>
                        <div style="background: #0d1117; padding: 15px; border-radius: 6px; font-family: monospace; font-size: 1.15rem; color: #8b949e;">
                            Recall = (Area of Overlap) / (Area of SAM Ground Truth Mask)
                        </div>
                    </div>
                </div>

                <!-- Right Column: Visual Example -->
                <div style="flex: 1.5; min-width: 500px;">
                    <h3 style="margin-top: 0; color: #c9d1d9; font-size: 1.8rem; border-bottom: 1px solid #30363d; padding-bottom: 10px;">Visual Calculation Example <span style="color: #8b949e; font-size: 1.1rem; font-weight: normal;">(Prompt: Blue cat, yellow sofa | Concept: "animal")</span></h3>
                    
                    <div style="display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 15px; margin-top: 20px;">
                        <div>
                            <span style="display: block; color: #8b949e; margin-bottom: 8px; font-size: 1.15rem;">1. Concept Mask (Pred)</span>
                            <img src="{concept_mask_src}" style="width: 100%; border-radius: 8px; border: 1px solid #30363d;" alt="Concept Mask">
                        </div>
                        <div>
                            <span style="display: block; color: #8b949e; margin-bottom: 8px; font-size: 1.15rem;">2. SAM Mask (Truth)</span>
                            <img src="{sam_mask_src}" style="width: 100%; border-radius: 8px; border: 1px solid #30363d;" alt="SAM Mask">
                        </div>
                        <div>
                            <span style="display: block; color: #8b949e; margin-bottom: 8px; font-size: 1.15rem;">3. Overlap Diagnostic</span>
                            <img src="{overlap_vis_src}" style="width: 100%; border-radius: 8px; border: 1px solid #30363d;" alt="Overlap Diagnostic">
                        </div>
                    </div>
                    
                    <div style="background: #0d1117; padding: 20px; border-radius: 6px; border: 1px solid #30363d; margin-top: 20px;">
                        <span style="color: #c9d1d9; font-size: 1.25rem; font-weight: bold; display: block; margin-bottom: 15px;">Visualizing the Overlap</span>
                        <ul style="color: #8b949e; font-size: 1.15rem; line-height: 1.6; padding-left: 20px; margin: 0;">
                            <li><strong style="color: #3fb950;">Green pixels (True Positives):</strong> Concept Mask AND SAM Mask overlap perfectly.</li>
                            <li><strong style="color: #58a6ff;">Blue pixels (False Positives):</strong> The Concept Mask highlighted this, but SAM says it isn't part of the object. (Hurts Precision).</li>
                            <li><strong style="color: #f85149;">Red pixels (False Negatives):</strong> SAM says this is part of the object, but the Concept Mask completely missed it. (Hurts Recall/Coverage).</li>
                        </ul>
                    </div>
                </div>
            </div>
        </div>
    """


def get_html_header_nav(title):
    """Return the HTML for the top navigation bar."""
    return f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
    <script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        {get_css()}
    </style>
</head>
<body>
    <div class="header-nav">
        <h1>{title}</h1>
        <div class="tabs">
            <button class="tab-btn active" onclick="switchTab('overview')">Overview</button>
            <button class="tab-btn" onclick="switchTab('discussion')">Discussion</button>
            <button class="tab-btn" onclick="switchTab('gallery')">Gallery</button>
            <button class="tab-btn" onclick="switchTab('grids')">Grids</button>
            <button class="tab-btn" onclick="switchTab('metrics')">Metrics</button>
            <button class="tab-btn" onclick="switchTab('implementation')">Implementation</button>
            <button class="tab-btn" onclick="switchTab('tokenization')">Tokenization</button>
        </div>
    </div>
    """


def get_gallery_navigator(toc_data):
    """Return a ``<select>`` dropdown for jumping to gallery groups/cases."""
    options = '<option value="">Jump to case…</option>'
    for group in toc_data:
        gname = group["group_name"]
        options += f'<optgroup label="{gname}">'
        for case in group["cases"]:
            cname = case["case_name"]
            options += f'<option value="{cname.lower()}">{cname}</option>'
        options += "</optgroup>"
    return f'<select id="gallery-nav" class="gallery-nav-select" onchange="jumpToGallerySection(this.value)">{options}</select>'


def get_explanation_block():
    """Return the welcome / explanation block shown on the Findings tab."""
    return """
                <div class="explanation-block" style="margin-bottom: 30px; padding: 20px; background: rgba(88, 166, 255, 0.05); border: 1px solid #30363d; border-radius: 8px; border-left: 4px solid #58a6ff;">
                    <h2 style="margin-top: 0; color: #c9d1d9; font-size: 1.3rem;">Welcome to the Concept Analysis Evaluation Dashboard</h2>
                    <p style="color: #8b949e; margin-bottom: 15px;">
                        This webpage presents a comprehensive evaluation of the <b>Concept Attention</b> mechanism, visualizing how well it localizes various concepts—ranging from concrete objects and colors to textures, spatial relations, and abstract ideas—within generated images.
                        The 1024x1024 pixel images are generated using the Flux1.schnell model in 4 inference steps, and the concept heatmaps are extracted from layers 16, 17, and 18 of the double stream attention blocks.
                    </p>
                    <div style="background: rgba(210, 153, 34, 0.1); border: 1px solid #d29922; border-radius: 6px; padding: 12px; margin-bottom: 15px; color: #d29922; font-size: 0.9rem;">
                        <strong>Technical Note:</strong> For concepts consisting of multiple text tokens (as determined by the T5 tokenizer), <strong>only the first token's attention map</strong> is currently used for localization. 
                        Note that the special character <code style="background: #161b22; padding: 2px 4px; border-radius: 3px;">\\u2581</code> (often rendered as <code style="background: #161b22; padding: 2px 4px; border-radius: 3px;">_</code>) is <strong>not considered a real token</strong> and is ignored during count calculations.
                    </div>
                    <ul style="color: #8b949e; margin-bottom: 15px; padding-left: 20px;">
                        <li style="margin-bottom: 8px;"><strong style="color: #c9d1d9;">Overview:</strong> Displays a high-level quantitative overview followed by key highlighted cases based on the evaluation findings.</li>
                        <li style="margin-bottom: 8px;"><strong style="color: #c9d1d9;">Discussion:</strong> A discussion highlighting key conclusions on when concept attention performs optimally.</li>
                        <li style="margin-bottom: 8px;"><strong style="color: #c9d1d9;">Gallery:</strong> A hierarchical browser to explore all generated images, prompts, and seeds. Expand concept sets to compare primary masked images, raw attention maps, and SAM segments side-by-side.</li>
                        <li style="margin-bottom: 8px;"><strong style="color: #c9d1d9;">Grids:</strong> A dense visual layout designed for qualitative assessment of attention heatmaps across different experimental runs.</li>
                        <li style="margin-bottom: 8px;"><strong style="color: #c9d1d9;">Tokenization:</strong> Illustrates how each concept is split into text tokens by the T5 encoder. This is critical for understanding localization, as multi-token concepts may exhibit different attention behaviors compared to single-token ones.</li>
                    </ul>
                    <p style="color: #8b949e; font-size: 0.9rem; margin-bottom: 0;">
                        <i>Tip: Use the tabs above to navigate between views, and expand the "Concept Sets" or "View Alternate Heatmaps" details in the cases below for deeper visual analysis.</i>
                    </p>
                </div>"""

def get_discussion_tab_html():
    return """
        <!-- DISCUSSION TAB -->
        <div id="discussion" class="tab-content" style="max-width: 1200px; margin: 0 auto; padding-top: 40px;">
            <h2 style="margin-top:0; font-size: 2.8rem; color:#c9d1d9; border-bottom: 1px solid #30363d; padding-bottom: 20px; margin-bottom: 40px;">Project Discussion & Outlook</h2>
            
            <!-- METHODOLOGY PANEL -->
            <div style="background: #161b22; border: 1px solid #30363d; border-radius: 12px; padding: 35px; margin-bottom: 40px; box-shadow: 0 10px 30px rgba(0,0,0,0.3);">
                <h3 style="margin-top: 0; color: #58a6ff; font-size: 1.8rem; display: flex; align-items: center; gap: 15px;">
                    <span style="background: #58a6ff; color: #0d1117; width: 35px; height: 35px; display: flex; justify-content: center; align-items: center; border-radius: 50%; font-size: 1.2rem;">1</span>
                    Establishing "Ground Truth" via SAM
                </h3>
                <div style="margin-top: 20px; color: #8b949e; font-size: 1.15rem; line-height: 1.8;">
                    <p>To quantitatively evaluate the <i>Concept Attention</i> heatmaps, I developed a heuristic to map concept tokens to <b>Segment Anything Model (SAM)</b> masks. By comparing the attention peaks against these segments, we can calculate metrics like <b>IoU (Intersection over Union)</b>.</p>
                    
                    <ul style="padding-left: 25px; margin-top: 15px;">
                        <li style="margin-bottom: 15px;">
                            <strong style="color: #c9d1d9;">Success in Object Classes:</strong> This approach works remarkably well for distinct, tangible objects (e.g., "blue cat" vs "sofa"). The high alignment scores suggest concept attention effectively locks onto physical boundaries.
                        </li>
                        <li style="margin-bottom: 15px;">
                            <strong style="color: #c9d1d9;">Failure with Spatial / Abstract Concepts:</strong> For concepts like "top" or "transparent", the heuristic often breaks down. SAM is inherently <i>object-centric</i>, making it difficult to generate a meaningful "ground truth" for non-object regions.
                        </li>
                        <li style="margin-bottom: 15px;">
                            <strong style="color: #c9d1d9;">Semantic Misalignment:</strong> In several cases, we observed instances where the IoU score was numerically high, but the assigned semantic label was logically incorrect. This suggests the attention mechanism might be "looking" at the right place for the wrong reason.
                        </li>
                    </ul>
                </div>
            </div>

            <!-- OUTLOOK PANEL -->
            <div style="background: #1d212b; border: 1px solid #388bfd66; border-radius: 12px; padding: 35px; margin-bottom: 40px; box-shadow: 0 10px 30px rgba(88, 166, 255, 0.1);">
                <h3 style="margin-top: 0; color: #79c0ff; font-size: 1.8rem; display: flex; align-items: center; gap: 15px;">
                    <span style="background: #79c0ff; color: #0d1117; width: 35px; height: 35px; display: flex; justify-content: center; align-items: center; border-radius: 50%; font-size: 1.2rem;">2</span>
                    Outlook: Future Research Paths
                </h3>
                
                <div style="margin-top: 25px; background: #0d1117; padding: 25px; border-radius: 10px; border: 1px solid #30363d; text-align: center; margin-bottom: 30px;">
                    <p style="color: #8b949e; margin-bottom: 15px; font-family: monospace;">Core Attention Formula (Softmax over Concept Dimension):</p>
                    <div style="font-size: 1.8rem; color: #c9d1d9;">
                        $$\phi(o_x, o_c) = \text{softmax}(o_x o_c^T)$$
                    </div>
                </div>

                <div style="color: #8b949e; font-size: 1.1rem; line-height: 1.8;">
                    <!-- POINT A -->
                    <div style="margin-bottom: 30px; border-left: 3px solid #79c0ff; padding-left: 20px;">
                        <h4 style="color: #c9d1d9; font-size: 1.3rem; margin-top: 0;">A. Reversing Mutual Exclusivity</h4>
                        <p>Currently, the softmax is applied across concepts for each patch (forcing concepts to compete for space). By swapping the cross-product to focus on the <i>spatial</i> dimension, we could identify the most relevant patches for every concept individually.</p>
                        <p style="font-style: italic; font-size: 0.95rem; color: #58a6ff55;">Suggested Experiment: Compare heatmaps generated with Softmax(Concepts) vs. Softmax(Pixels) to measure separation quality.</p>
                    </div>

                    <!-- POINT B -->
                    <div style="margin-bottom: 30px; border-left: 3px solid #79c0ff; padding-left: 20px;">
                        <h4 style="color: #c9d1d9; font-size: 1.3rem; margin-top: 0;">B. Breaking the Single-Token Constraint</h4>
                        <p>Concepts are currently limited to single tokens. This fails for complex prompts (e.g., "owl" vs "beak" vs "feathers"). Aggregating attention from multiple tokens into a single semantic heatmap represents a massive opportunity for improvement.</p>
                        <p style="font-style: italic; font-size: 0.95rem; color: #58a6ff55;">Suggested Experiment: Test various pooling strategies (Mean, Max, or Attention-weighted) for multi-token concepts.</p>
                    </div>

                    <!-- POINT C -->
                    <div style="margin-bottom: 10px; border-left: 3px solid #79c0ff; padding-left: 20px;">
                        <h4 style="color: #c9d1d9; font-size: 1.3rem; margin-top: 0;">C. Solving the "Fallback Problem" (Sink Token)</h4>
                        <p>Because the attention must sum to 100%, semantically nonsensical concepts often receive artificial boosts in attention. Introducing a "Sink Token" (a background junk token) would allow for low-confidence regions to be absorbed mathematically rather than assigned randomly.</p>
                        <p style="font-style: italic; font-size: 0.95rem; color: #58a6ff55;">Suggested Experiment: Inject a learnable or fixed "neutral" embedding into the concept set and observe if it "soaks" background noise.</p>
                    </div>
                </div>
            </div>

            <!-- KEY FINDINGS SECTION (ORIGINAL) -->
            <div style="background: #161b22; border: 1px solid #30363d; border-radius: 12px; padding: 35px; margin-bottom: 50px;">
                <h3 style="margin-top: 0; color: #d29922; font-size: 1.6rem; border-bottom: 1px solid #30363d; padding-bottom: 10px;">Summary of Findings</h3>
                <ul style="color: #8b949e; font-size: 1.2rem; line-height: 1.7; padding-left: 25px; margin-top: 20px;">
                    <li><strong style="color: #c9d1d9;">Precision requires Priors:</strong> the model performs optimally when the set of concepts is pre-defined and aligned with the prompt.</li>
                    <li><strong style="color: #c9d1d9;">Sensitivity to Generality:</strong> General concepts (e.g., "something", "background") tend to create diffuse attention maps that lack precision.</li>
                    <li>In the original authors' quantitative evaluation, handcrafted concept vocabularies were used for every image to achieve high-quality results.</li>
                </ul>
            </div>
        </div>
    """


def get_statistics_section_html():
    """Return the HTML for the single/multi-token statistics section (chart canvases)."""
    return """
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


def get_umap_section_html():
    """Return the HTML for the UMAP scatter-plot section."""
    return """
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


def get_failure_tab_header(num_hallucination, num_missed):
    """Return the header block for the Failure Analysis tab."""
    return f"""
            <div id="failure" class="tab-content">
                <div style="margin-bottom: 30px; padding: 20px; background: rgba(248, 81, 73, 0.05); border: 1px solid #30363d; border-radius: 8px; border-left: 4px solid #f85149;">
                    <h2 style="margin-top: 0; color: #f85149;">Failure Case Analysis</h2>
                    <p style="color: #8b949e;">Cases where Concept Attention significantly diverges from SAM Ground Truth.</p>
                    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-top: 15px;">
                        <div style="background: rgba(248, 81, 73, 0.1); padding: 12px; border-radius: 6px; border: 1px solid #f8514933;">
                            <strong style="color: #f85149;">Hallucinations (Low Precision)</strong><br>
                            <span style="color: #8b949e; font-size: 0.85rem;">Attention activates where no matching SAM segment exists.</span>
                        </div>
                        <div style="background: rgba(210, 153, 34, 0.1); padding: 12px; border-radius: 6px; border: 1px solid #d2992233;">
                            <strong style="color: #d29922;">Missed Localizations (Low Recall)</strong><br>
                            <span style="color: #8b949e; font-size: 0.85rem;">SAM finds the object but Attention fails to activate on it.</span>
                        </div>
                    </div>
                </div>
                
                <h3 style="color: #f85149; border-bottom: 1px solid #f8514933; padding-bottom: 10px;">Hallucinations (showing worst {min(10, num_hallucination)} of {num_hallucination})</h3>
    """


def get_javascript(single_token_averages, multi_token_averages, umap_js_data):
    """Return the full ``<script>`` block as a string."""
    return (
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
                            `  Std: \\u00b1${stdVal.toFixed(3)}`,
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

        // ── Gallery navigator dropdown ──
        function jumpToGallerySection(val) {
            if (!val) return;
            const searchInput = document.getElementById('search-input');
            if (searchInput) {
                searchInput.value = val;
                searchInput.dispatchEvent(new Event('input', { bubbles: true }));
            }
            // Reset dropdown to placeholder
            document.getElementById('gallery-nav').selectedIndex = 0;
            
            // Scroll to top of content area to see the result
            const container = document.querySelector('.content-area');
            if (container) {
                container.scrollTo({top: 0, behavior: 'smooth'});
            }
        }

        // ── Concept-set tabs ──
        function switchConceptTab(caseId, idx) {
            const container = document.getElementById(caseId + '-tabs');
            if (!container) return;
            container.querySelectorAll('.concept-tab-btn').forEach(b => b.classList.remove('active'));
            container.querySelectorAll('.concept-tab-btn')[idx].classList.add('active');

            const parent = container.parentElement;
            parent.querySelectorAll('.concept-tab-panel').forEach(p => p.classList.remove('active'));
            parent.querySelectorAll('.concept-tab-panel')[idx].classList.add('active');
        }

        function switchTab(tabId) {
            // Update buttons
            document.querySelectorAll('.tab-btn').forEach(btn => btn.classList.remove('active'));
            event.target.classList.add('active');
            
            // Update content visibility
            document.querySelectorAll('.tab-content').forEach(content => content.classList.remove('active'));
            document.getElementById(tabId).classList.add('active');
        }
        
        // Search and Category filtering for Gallery
        const searchInput = document.getElementById('search-input');
        const sections = document.querySelectorAll('.group-section');
        const categoryButtons = document.querySelectorAll('.cat-btn');
        let activeCategories = new Set();

        function updateGalleryFilter() {
            const query = searchInput.value.toLowerCase().trim();
            const searchWords = query.split(/\\s+/).filter(word => word.length > 0);
            
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

        // ── Image size slider ──
        const imgSlider = document.getElementById('img-size-slider');
        const imgSliderVal = document.getElementById('img-size-val');
        if (imgSlider) {
            imgSlider.addEventListener('input', (e) => {
                const px = e.target.value;
                document.documentElement.style.setProperty('--gallery-img-size', px + 'px');
                imgSliderVal.textContent = px + 'px';
            });
        }

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
            const searchWords = query.split(/\\s+/).filter(word => word.length > 0);
            
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
