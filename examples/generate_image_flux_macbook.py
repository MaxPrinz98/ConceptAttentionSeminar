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

    print(f"Number of double blocks: {len(pipeline.flux_generator.model.double_blocks)}")


    prompt = "An Owl flying over a university campus"
    concepts = ["Owl", "university" , "sky", "feathers", "bird", "buldings", "ground", "trees"]
    pipeline_output = pipeline.generate_image(
        prompt=prompt,
        concepts=concepts,
        width=1024,  # Start with 512 to test if your memory can handle it
        height=1024,
        layer_indices=[16, 17, 18],
        num_inference_steps=4,
        timesteps=list(range(0, 4))
    )

    image = pipeline_output.image
    concept_heatmaps = pipeline_output.concept_heatmaps

    os.makedirs("results/flux", exist_ok=True)
    image.save("results/flux/image.png")
    
    for concept, heatmap in zip(concepts, concept_heatmaps):
        heatmap.save(f"results/flux/{concept}.png")

    print("Success! Results saved to results/flux/")