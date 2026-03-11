import json
from concept_attention.flux.image_generator import FluxGenerator

def main():
    generator = FluxGenerator(model_name="flux-schnell", device="cpu", offload=True)
    t5_tokenizer = generator.t5.tokenizer
    clip_tokenizer = generator.clip.tokenizer

    words = ["violence", "multicolor", "human face", "astronaut", "blue_cat", "picket fence"]
    results = {}
    for word in words:
        results[word] = {
            "t5_tokens": t5_tokenizer.tokenize(word),
            "t5_ids": t5_tokenizer.encode(word),
            "clip_tokens": clip_tokenizer.tokenize(word),
            "clip_ids": clip_tokenizer.encode(word),
        }
    with open("token_results.json", "w") as f:
        json.dump(results, f, indent=4)

if __name__ == "__main__":
    main()
