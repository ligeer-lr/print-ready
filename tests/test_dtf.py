import io
import zipfile

import numpy as np
import pikepdf
import pytest
from fastapi.testclient import TestClient
from PIL import Image, ImageDraw

from app.dtf import process_dtf
from app.main import app
from app.processor import ProcessingError

client = TestClient(app)


def artwork_on_flat_bg(size=(400, 500), bg=(235, 228, 205)) -> bytes:
    img = Image.new("RGB", size, bg)
    d = ImageDraw.Draw(img)
    d.ellipse((100, 120, 300, 380), fill=(200, 30, 40))
    d.ellipse((170, 220, 230, 280), fill=bg)  # same colour as bg, but enclosed
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return buf.getvalue()


def unzip(blob: bytes) -> dict[str, bytes]:
    z = zipfile.ZipFile(io.BytesIO(blob))
    return {n: z.read(n) for n in z.namelist()}


def test_bundle_contents_and_size():
    blob, rep = process_dtf(artwork_on_flat_bg(), width_cm=10, bg_method="color")
    files = unzip(blob)
    assert set(files) == {"artwork.png", "white-plate.png", "artwork-dtf.pdf"}
    art = Image.open(io.BytesIO(files["artwork.png"]))
    assert art.mode == "RGBA"
    assert art.size == rep.pixels
    assert rep.pixels[0] == round(10 / 2.54 * 300)
    assert art.info["dpi"][0] == pytest.approx(300, abs=1)


def test_flat_background_removed_but_enclosed_hole_kept():
    blob, rep = process_dtf(artwork_on_flat_bg(), width_cm=10, bg_method="color")
    art = np.asarray(Image.open(io.BytesIO(unzip(blob)["artwork.png"])))
    alpha = art[..., 3]
    assert set(np.unique(alpha)) <= {0, 255}  # no half-transparent edges
    h, w = alpha.shape
    assert alpha[0, 0] == 0  # corner is background
    assert alpha[h // 2, w // 2] == 255  # enclosed same-colour ellipse stays
    assert rep.background_removed is True


def test_white_plate_is_choked():
    blob, rep = process_dtf(artwork_on_flat_bg(), width_cm=10, bg_method="color", choke_mm=0.5)
    files = unzip(blob)
    art = np.asarray(Image.open(io.BytesIO(files["artwork.png"])))[..., 3] > 0
    plate = np.asarray(Image.open(io.BytesIO(files["white-plate.png"]))) < 128
    assert plate.sum() < art.sum()
    assert not (plate & ~art).any()  # white never outside the colour


def test_pdf_has_white_spot_and_no_softmask():
    blob, _ = process_dtf(artwork_on_flat_bg(), width_cm=10, bg_method="color")
    pdf = pikepdf.open(io.BytesIO(unzip(blob)["artwork-dtf.pdf"]))
    page = pdf.pages[0]
    cs = page.Resources.ColorSpace.CS0
    assert cs[0] == "/Separation" and cs[1] == "/White"
    im = page.Resources.XObject.Im0
    assert "/SMask" not in im and "/Mask" in im
    assert str(pdf.docinfo.GTS_PDFXVersion) == "PDF/X-1a:2001"
    assert pdf.pdf_version == "1.3"


def test_transparent_input_keeps_alpha():
    img = Image.new("RGBA", (200, 200), (0, 0, 0, 0))
    ImageDraw.Draw(img).ellipse((50, 50, 150, 150), fill=(10, 120, 200, 255))
    buf = io.BytesIO()
    img.save(buf, "PNG")
    blob, rep = process_dtf(buf.getvalue(), width_cm=8)
    assert rep.background_removed is False
    assert rep.width_cm == 8


def test_blank_image_rejected_and_bad_params():
    flat = Image.new("RGB", (100, 100), (255, 255, 255))
    buf = io.BytesIO()
    flat.save(buf, "PNG")
    with pytest.raises(ProcessingError):
        process_dtf(buf.getvalue(), bg_method="color")
    with pytest.raises(ProcessingError):
        process_dtf(artwork_on_flat_bg(), width_cm=500)
    with pytest.raises(ProcessingError):
        process_dtf(b"junk")


def test_api_dtf():
    r = client.post(
        "/api/dtf",
        files={"file": ("a.png", artwork_on_flat_bg(), "image/png")},
        data={"width_cm": "12", "bg_method": "color"},
    )
    assert r.status_code == 200
    assert r.headers["content-type"] == "application/zip"
    assert "artwork-dtf.pdf" in zipfile.ZipFile(io.BytesIO(r.content)).namelist()
    bad = client.post("/api/dtf", files={"file": ("a.png", b"junk", "image/png")})
    assert bad.status_code == 422


def _two_tone_scene():
    """Background has two flat colours (left blue, right green) so auto keying sees a non-uniform border."""
    img = Image.new("RGB", (300, 300), (40, 90, 200))
    d = ImageDraw.Draw(img)
    d.rectangle((150, 0, 300, 300), fill=(40, 170, 80))
    d.ellipse((90, 90, 210, 210), fill=(210, 40, 40))
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return buf.getvalue()


def test_picked_colours_are_removed():
    picks = "[[0.02, 0.5], [0.98, 0.5]]"  # one click on each background colour
    blob, rep = process_dtf(_two_tone_scene(), width_cm=10, picks=picks, tolerance=30)
    alpha = np.asarray(Image.open(io.BytesIO(unzip(blob)["artwork.png"])))[..., 3]
    h, w = alpha.shape
    assert alpha[h // 2, w // 2] == 255
    assert alpha[2, 2] == 0 and alpha[2, w - 3] == 0
    assert 0.6 < (alpha > 0).mean() < 0.95
    assert rep.background_removed is True


def test_one_pick_removes_only_that_colour():
    blob, _ = process_dtf(_two_tone_scene(), width_cm=10, picks="[[0.02, 0.5]]")
    alpha = np.asarray(Image.open(io.BytesIO(unzip(blob)["artwork.png"])))[..., 3]
    h, w = alpha.shape
    assert alpha[2, 2] == 0               # blue background gone (corner of the cropped result)
    assert alpha[h // 2, w - 3] == 255    # green side stays


def test_scope_everywhere_removes_inner_pockets():
    img = Image.new("RGB", (300, 300), (240, 240, 240))
    d = ImageDraw.Draw(img)
    d.ellipse((60, 60, 240, 240), fill=(200, 30, 30))
    d.ellipse((130, 130, 170, 170), fill=(240, 240, 240))  # enclosed pocket, same as bg
    buf = io.BytesIO()
    img.save(buf, "PNG")
    kept, _ = process_dtf(buf.getvalue(), width_cm=10, picks="[[0.01, 0.01]]", scope="contiguous")
    gone, _ = process_dtf(buf.getvalue(), width_cm=10, picks="[[0.01, 0.01]]", scope="everywhere")
    a_kept = np.asarray(Image.open(io.BytesIO(unzip(kept)["artwork.png"])))[..., 3]
    a_gone = np.asarray(Image.open(io.BytesIO(unzip(gone)["artwork.png"])))[..., 3]
    c = a_kept.shape[0] // 2, a_kept.shape[1] // 2
    assert a_kept[c] == 255 and a_gone[c] == 0


def test_bad_picks_rejected():
    for bad in ("not json", "[[2, 2]]", "[[0.5]]"):
        with pytest.raises(ProcessingError):
            process_dtf(_two_tone_scene(), width_cm=10, picks=bad)
    with pytest.raises(ProcessingError):
        process_dtf(_two_tone_scene(), width_cm=10, picks="[[0.1,0.1]]", tolerance=500)
    with pytest.raises(ProcessingError):
        process_dtf(_two_tone_scene(), width_cm=10, picks="[[0.1,0.1]]", scope="x")


def test_preview_endpoint_with_picks():
    r = client.post(
        "/api/dtf/preview",
        files={"file": ("s.png", _two_tone_scene(), "image/png")},
        data={"picks": "[[0.02, 0.5], [0.98, 0.5]]", "tolerance": "30"},
    )
    assert r.status_code == 200
    img = np.asarray(Image.open(io.BytesIO(r.content)))
    assert img[150, 150, 3] == 255 and img[150, 3, 3] == 0 and img[150, 296, 3] == 0
