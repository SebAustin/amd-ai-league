"""Turn any input image (PNG, JPEG, TIFF; any size, mode or EXIF rotation) into the views the
vision model reads: the upright original and a contrast-boosted copy for dark or glare shots."""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path

log = logging.getLogger(__name__)

# Every view is letterboxed onto one fixed canvas. The GPU compiles kernels once per input
# shape (~15-25 s each on a Radeon PRO): with arbitrary sizes every new image would pay that
# inside its 30 s budget; with one shape it is paid once, at server warm-up.
CANVAS = 1024
PADDING = (114, 114, 114)  # neutral grey: neither a white plate nor a black night sky
CUTOFF_PERCENT = 2  # autocontrast ignores the darkest/brightest 2% (glare, black borders)


def letterbox(img):
    """``img`` scaled to fit CANVAS x CANVAS (up or down, aspect kept), centred on grey."""
    from PIL import Image

    width, height = img.size
    scale = CANVAS / max(width, height)
    size = (max(1, round(width * scale)), max(1, round(height * scale)))
    canvas = Image.new("RGB", (CANVAS, CANVAS), PADDING)
    canvas.paste(img.resize(size, Image.Resampling.LANCZOS),
                 ((CANVAS - size[0]) // 2, (CANVAS - size[1]) // 2))
    return canvas


def _rgb(img):
    from PIL import Image

    if img.mode in ("RGBA", "LA", "P"):  # transparent areas become white, not black
        rgba = img.convert("RGBA")
        canvas = Image.new("RGB", rgba.size, "white")
        canvas.paste(rgba, mask=rgba.getchannel("A"))
        return canvas
    return img.convert("RGB")


def prepare(image: Path, scratch: Path) -> list[Path]:
    """The views to read, as PNG files in ``scratch``; [] when the file cannot be decoded."""
    try:
        from PIL import Image, ImageOps

        with Image.open(image) as raw:
            raw.seek(0)  # multi-page TIFF: the first page
            upright = _rgb(ImageOps.exif_transpose(raw))
            boosted = ImageOps.autocontrast(ImageOps.grayscale(upright), cutoff=CUTOFF_PERCENT)
            upright, boosted = letterbox(upright), letterbox(boosted.convert("RGB"))
        scratch.mkdir(parents=True, exist_ok=True)
        stem = hashlib.sha1(str(image.resolve()).encode()).hexdigest()[:16]
        views = [scratch / f"{stem}-original.png", scratch / f"{stem}-contrast.png"]
        upright.save(views[0])
        boosted.save(views[1])
        return views
    except Exception:  # any undecodable input is an empty answer, not a crash
        log.warning("cannot read image %s", image, exc_info=True)
        return []
