import os

from render_html import generate_gallery

if __name__ == "__main__":
    results_dir = "results/object_analysis"
    rel_html = os.path.join(results_dir, "evaluation_gallery.html")
    generate_gallery(results_dir, rel_html, mode="relative")

    full_html = os.path.join(results_dir, "evaluation_gallery_full.html")
    generate_gallery(results_dir, full_html, mode="base64")
