import os
import json
import torch
import numpy as np
import umap
from tqdm import tqdm
from pathlib import Path
from concept_attention.flux.image_generator import FluxGenerator


def get_unique_concepts(root_dir):
    concepts = set()
    for path in Path(root_dir).rglob("metadata.json"):
        with open(path, "r") as f:
            data = json.load(f)
            concepts.update(data.get("concepts", []))
    return sorted(list(concepts))


def extract_embeddings(concepts):
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Loading FluxGenerator to access T5 on {device}...")
    # Use offload=True to keep memory usage low
    generator = FluxGenerator(model_name="flux-schnell", device=device, offload=True)

    tokenizer = generator.t5.tokenizer
    model = generator.t5.hf_module
    model.eval()

    embeddings = []
    print(f"Extracting embeddings for {len(concepts)} concepts...")
    for concept in tqdm(concepts):
        # HFEmbedder logic usually handles padding, but here we just want the embedding
        inputs = tokenizer(concept, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
            # Use the same pooling as HFEmbedder: last_hidden_state[0, 0, :]
            emb = outputs.last_hidden_state[0, 0, :].cpu().float().numpy()
            embeddings.append(emb)
    return np.array(embeddings)


def main():
    root_dir = "results/object_analysis"
    output_path = os.path.join(root_dir, "concept_umap.json")
    concepts = get_unique_concepts(root_dir)
    if not concepts:
        print("No concepts found.")
        return

    embeddings = extract_embeddings(concepts)
    print(f"Running UMAP on {len(concepts)} concepts...")
    reducer = umap.UMAP(
        n_neighbors=10,  # Reduced from 15 to focus on local structure
        min_dist=0.01,  # Reduced from 0.1 to allow tighter packing
        n_components=2,
        random_state=42,
    )
    embedding_2d = reducer.fit_transform(embeddings)

    result = {concept: embedding_2d[i].tolist() for i, concept in enumerate(concepts)}
    with open(output_path, "w") as f:
        json.dump(result, f, indent=4)
    print(f"Saved UMAP to {output_path}")


if __name__ == "__main__":
    main()
