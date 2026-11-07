from PIL import Image

from amdleague.m2.images import CANVAS, prepare


def _save(path, size=(400, 200), color=(200, 200, 200), **kw):
    Image.new("RGB", size, color).save(path, **kw)
    return path


def test_png_jpeg_and_tiff_become_rgb_png_views(tmp_path):
    for name in ("a.png", "b.jpg", "c.tif"):
        views = prepare(_save(tmp_path / name), tmp_path / "scratch")
        assert len(views) == 2  # original + contrast-boosted
        for view in views:
            with Image.open(view) as img:
                assert img.format == "PNG" and img.mode == "RGB"


def test_every_view_has_the_same_canvas_size(tmp_path):
    """One input shape for the vision model: a new shape costs ~20 s of GPU kernel compilation."""
    for name, size in (("big.png", (6000, 3000)), ("small.png", (120, 60)),
                       ("tall.jpg", (300, 900)), ("square.tif", (610, 610))):
        for view in prepare(_save(tmp_path / name, size), tmp_path / "s"):
            with Image.open(view) as img:
                assert img.size == (CANVAS, CANVAS)


def test_letterbox_keeps_the_aspect_ratio(tmp_path):
    Image.new("RGB", (800, 200), (0, 0, 0)).save(tmp_path / "plate.png")  # black 4:1 strip
    with Image.open(prepare(tmp_path / "plate.png", tmp_path / "s")[0]) as img:
        dark_rows = [y for y in range(CANVAS) if img.getpixel((CANVAS // 2, y))[0] < 50]
    assert abs(len(dark_rows) - CANVAS // 4) <= 2  # content is CANVAS wide, CANVAS/4 tall


def test_exif_rotation_is_applied(tmp_path):
    img = Image.new("RGB", (400, 200), (10, 10, 10))
    exif = img.getexif()
    exif[0x0112] = 6  # rotated 90° clockwise
    img.save(tmp_path / "phone.jpg", exif=exif)
    with Image.open(prepare(tmp_path / "phone.jpg", tmp_path / "s")[0]) as out:
        dark_cols = [x for x in range(CANVAS) if out.getpixel((x, CANVAS // 2))[0] < 50]
    assert len(dark_cols) < CANVAS * 0.6  # upright: now taller than wide


def test_rgba_and_greyscale_inputs_work(tmp_path):
    Image.new("RGBA", (300, 300), (0, 0, 0, 0)).save(tmp_path / "t.png")
    Image.new("L", (300, 300), 50).save(tmp_path / "g.tif")
    assert prepare(tmp_path / "t.png", tmp_path / "s")
    assert prepare(tmp_path / "g.tif", tmp_path / "s")


def test_unreadable_file_gives_no_views(tmp_path):
    (tmp_path / "bad.png").write_bytes(b"not an image")
    assert prepare(tmp_path / "bad.png", tmp_path / "s") == []
    assert prepare(tmp_path / "missing.png", tmp_path / "s") == []


def test_server_warmup_reads_an_image_of_exactly_the_canvas_size(tmp_path):
    from amdleague.common.model_server import WARMUP_IMAGE_SIDE, warmup_messages

    assert WARMUP_IMAGE_SIDE == CANVAS
    text_only = warmup_messages(False, tmp_path)
    vision = warmup_messages(True, tmp_path)
    assert len(text_only) == 1 and len(vision) == 2
    image_part = vision[1][-1]["content"][0]
    with Image.open(image_part["path"]) as img:
        assert img.size == (CANVAS, CANVAS)
