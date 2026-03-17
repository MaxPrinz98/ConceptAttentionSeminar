# Seminar Project: Automated Evaluation of Concept Attention

## Technical Overview

The project implements an **automated evaluation pipeline** for Concept Attention in text-to-image models (specifically Flux). It bridges the gap between raw attention heatmaps and measurable semantic accuracy.

### 1. The Pipeline
- **Generation (Flux-schnell)**: Generates 1024x1024 images while extracting cross-attention maps for target concepts.
- **Processing**: Upscales maps to image resolution and applies thresholding to create concept-specific binary masks.
- **Ground Truth via SAM2**: Since we lack labels, we use the **Segment Anything Model (SAM2)** to generate high-quality segmentations of the image.
- **Metric Computation**: We greedily match concept masks to SAM segments to calculate **IoU, Precision, and Recall**. This quantifies how well the model's "internal attention" aligns with the actual "pixels" it generated.
- **Analysis**: Extracts T5 text embeddings to analyze the "Concept Space" via UMAP and identifies clusters of high/low performance.
---
### 2. The Directory structure
- **Hierarchical Storage**: `results/object_analysis/[category]/[name]/seed_[#]/set_[concepts]/`
- **Dashboad vs File System**: The dashboard (`evaluation_gallery.html`) acts as a lens into this structure, flattening the hierarchy for comparison while allowing "drill-down" into specific seeds.
- **Key Artifacts**:
    - `image.png`: The output.
    - `upscaled_heatmap_*.png`: Colormapped (Inferno) attention maps at 1024x1024.
    - `mask_*.png`: The binary decision of "where the model thinks it is".
    - `heatmap_segmentation.png`: The "internal world map" of everything the model attended to.

---
### 3. Implementation details of thresholding and segmentations
- **Relative Thresholding**:
    - Calculated as `cutoff = 0.5 * max(heatmap_value)`.
    - Using a relative threshold instead of an absolute one (e.g., fixed 0.1) handles concepts with different "confidence levels" equally.
- **Argmax Segmentation**:
    - Resolves overlaps where multiple concepts overlap in cross-attention.
    - We stack all concept heatmaps and add a constant **Background Layer (0.1)**.
    - `np.argmax` picks the "winner" for every pixel, creating a mutually exclusive semantic map.

---

### 4. Implementation details of SAM segmentation
- **100% Coverage**:
    - SAM is designed for object discovery and often leaves "gaps" between segments.
    - We calculate a **synthetic background segment** by inverting the union of all SAM masks (`~union`). This ensures every pixel is accounted for in the evaluation.
- **Greedy Composition (The Matching Algorithm)**:
    - We don't just pick *one* SAM segment; a concept might span multiple (e.g., "ears" and "tail" of a cat).
    - **Initialization**: The algorithm starts with whichever single SAM segment has the **highest individual IoU** with the concept mask.
    - **Strict Improvement (`>`)**:
        - In each step, we only add a segment if it **strictly increases** the total IoU (`temp_iou > best_iou`).
        - If a segment is entirely contained within the already selected ones, it is ignored (since IoU wouldn't change).
        - This ensures the smallest, most efficient set of segments is chosen to represent the concept.
- **Metrics Breakdown**:
    - **IoU**: Overall semantic alignment.
    - **Precision**: How well the attention is "contained" within a real object (avoids bleeding).
    - **Recall**: How much of the real object the attention "covers" (avoids missing parts).


---
### 5: High-Level Results (5 mins)
- **Aggregated Metrics**: Open the gallery and show the summary table. Highlight which categories performed best (e.g., distinct objects) vs worst (e.g., abstract attributes or tiny details).
- **Concept UMAP**: Switch to the UMAP view. Show how semantically similar concepts (like different colors or animals) cluster together and how their performance varies by cluster.
---
### Part 6: Qualitative Analysis & Failure Cases (10 mins)
- **The "Perfect" Match**: Pick an example with high IoU (e.g., "blue cat"). Show how the heatmaps perfectly align with the object.

- **Failure: Hallucinations**: Find a case where the concept has high attention but no corresponding object in the image (or vice versa).
- **Failure: Attribute Bleeding**: Show a case where a color concept "bleeds" into the background or adjacent objects.
---
### Part 7: Under the Hood (5 mins)
- **Mask Generation**: Click into a sample and show the "Implementation" tab. Explain the Argmax Segmentation—how we resolve overlapping attention maps into a single segmentation.
- **Metrics Breakdown**: Show the SAM segment matching logic (how we identify which part of the image the model *meant* to draw).

---
### Cool findings:
- multicolor_blocks
    - object concept dominate color, spatial, etc.
- blue_cat_yellow_sofa
    - since image patches the precision is a bit lower -> could use combination of concept attention and sam segments to get even better segmenattions (however sam is only working well on objects)
- 