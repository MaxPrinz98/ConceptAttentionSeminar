# Assumtion: Generate image and only give a single concept, etc. such that it can be verified if the sum of all concepts gives the whole picture again?
# Generate same image with prompt multiple times. check if synonyms in concepts give the same heatmap. and check what happens if there are synonyms in same concept array.


# For Report writing findings:
- Segmentation is bigger than the objects most of the time, since one patch = 16x16 pixels
- The Segmentation often is still noisy (maybe change threshold)
- If image generation fail, concept works correctly (dog with human face)