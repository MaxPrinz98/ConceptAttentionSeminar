import os
from pathlib import Path

from render_html import generate_gallery

if __name__ == "__main__":
    base_dir = Path(__file__).parent
    results_dir = base_dir.parent / "results/object_analysis"
    rel_html = results_dir / "evaluation_gallery.html"
    generate_gallery(results_dir, rel_html, mode="relative")

    full_html = results_dir / "evaluation_gallery_full.html"
    generate_gallery(results_dir, full_html, mode="base64")
