"""
Image encoding and path-resolution utilities.

Provides helpers for base64-encoding images, resolving image paths
relative to the HTML output file, and sanitising concept names for
use in filenames.
"""

import os
import base64
from pathlib import Path
from io import BytesIO
from PIL import Image


def image_to_base64(img_path, max_size=(512, 512)):
    """Convert an image to a base64-encoded data URI string."""
    if not os.path.exists(img_path):
        return ""
    try:
        img = Image.open(img_path)
        img.thumbnail(max_size)
        buffered = BytesIO()
        img.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        return f"data:image/png;base64,{img_str}"
    except Exception:
        return ""


def make_image_src_resolver(mode, output_html):
    """
    Return a closure that resolves an image path to an ``src`` attribute
    value, depending on the selected *mode* (``"base64"``, ``"absolute"``,
    or ``"relative"``).
    """

    def get_image_src(path):
        if not Path(path).exists():
            return ""
        if mode == "base64":
            return image_to_base64(path)
        elif mode == "absolute":
            return f"file://{Path(path).absolute()}"
        else:
            html_parent = Path(output_html).parent
            return os.path.relpath(path, html_parent)

    return get_image_src


def get_metric_class(val):
    """Return a CSS class name indicating metric quality."""
    if val >= 0.7:
        return "metric-good"
    if val < 0.4:
        return "metric-poor"
    return "metric-fair"


def safe_concept_name(concept):
    """Sanitise a concept string for safe use in filenames."""
    return (
        concept.replace(" ", "_")
        .replace("'", "")
        .replace('"', "")
        .replace("/", "-")
    )
