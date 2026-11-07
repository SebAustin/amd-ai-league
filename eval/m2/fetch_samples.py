"""Extract the 10 Match 2 sample images from the public brief into eval/m2/samples (git-ignored).

Usage: python eval/m2/fetch_samples.py [brief.pdf]   # downloads the brief when no path is given

The expected answers below are the brief's own; they grade our eval only and never reach app code.
"""

from __future__ import annotations

import json
import sys
import tempfile
import urllib.request
from pathlib import Path

BRIEF_URL = "https://drive.google.com/uc?export=download&id=117-uYCY_a82djUHCgzk7m6LkZI_NtHDD"
OUT = Path(__file__).parent / "samples"
# (format the brief says the sample comes in, expected answer, category)
SAMPLES = [
    ("png", "7ABC123", "US plate, clean"),
    ("png", "京A·12345", "Chinese plate, clean"),
    ("jpg", "JHT 2951", "US plate, angled"),
    ("png", "5XYZ891", "US plate, motion blur"),
    ("jpg", "沪B·88888", "Chinese plate, low light + glare"),
    ("png", "STOP", "stop sign, clean"),
    ("tif", "STOP", "stop sign, sensor noise"),
    ("jpg", "SPEED LIMIT 65", "speed limit sign"),
    ("png", "ROAD WORK AHEAD", "work zone sign, multi-line"),
    ("tif", "35", "advisory speed plaque"),
]


def main() -> int:
    from pypdf import PdfReader

    if len(sys.argv) > 1:
        pdf = Path(sys.argv[1])
    else:
        pdf = Path(tempfile.mkdtemp()) / "brief.pdf"
        urllib.request.urlretrieve(BRIEF_URL, pdf)  # fixed https URL
    images = [im.image for page in PdfReader(pdf).pages for im in page.images]
    if len(images) != len(SAMPLES):
        print(f"expected {len(SAMPLES)} images in the brief, found {len(images)}", file=sys.stderr)
        return 1
    OUT.mkdir(exist_ok=True)
    cases = []
    for n, (image, (fmt, answer, category)) in enumerate(zip(images, SAMPLES, strict=True), 1):
        name = f"image_{n:02d}.{fmt}"
        image.convert("RGB").save(OUT / name)
        cases.append({"image": name, "expected": answer, "category": category})
    (OUT / "samples.json").write_text(json.dumps(cases, ensure_ascii=False, indent=2))
    print(f"wrote {len(cases)} samples to {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
