"""
    Producing concept heatmaps for a generated image on Apple Silicon.
"""
import os
import torch
from concept_attention import ConceptAttentionFluxPipeline
#from huggingface_hub import login

# 1. Use your real token here (keep 'token=' to be safe)
#login(token='')

# 2. Define the device for Apple Silicon
device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {device}")

if __name__ == "__main__":
    # 3. Flux is EXTREMELY memory-heavy. 
    # Schnell usually needs ~24GB of RAM/VRAM. 
    # If your Mac has 8GB or 16GB, you might need to use float16.
    pipeline = ConceptAttentionFluxPipeline(
        model_name="flux-schnell",
        device=device
    )

    prompt = "A cat in a park on the grass by a tree"
    concepts = ["cat", "grass", "sky", "tree"]

    pipeline_output = pipeline.generate_image(
        prompt=prompt,
        concepts=concepts,
        width=1024,  # Start with 512 to test if your memory can handle it
        height=1024,
    )

    image = pipeline_output.image
    concept_heatmaps = pipeline_output.concept_heatmaps

    os.makedirs("results/flux", exist_ok=True)
    image.save("results/flux/image.png")
    
    for concept, heatmap in zip(concepts, concept_heatmaps):
        heatmap.save(f"results/flux/{concept}.png")

    print("Success! Results saved to results/flux/")