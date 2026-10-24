import io

import numpy as np
import pikepdf
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.main import app
from app.processor import MM_TO_PT, ProcessingError, process_image

client = TestClient(app)


def make_png(size=(600, 800), color=(255, 0, 0), mode="RGB") -> bytes:
    img = Image.new(mode, size, color)
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return buf.getvalue()


def box(arr):
    return [float(x) for x in arr]


def test_pdfx1a_structure():
    pdf_bytes, report = process_image(make_png(), preset="a4", bleed_mm=3, filename="x.png")
    pdf = pikepdf.open(io.BytesIO(pdf_bytes))
    assert pdf.pdf_version == "1.3"
    assert str(pdf.docinfo.GTS_PDFXVersion) == "PDF/X-1a:2001"
    assert str(pdf.docinfo.Trapped) == "/False"
    oi = pdf.Root.OutputIntents[0]
    assert oi.S == "/GTS_PDFX"
    assert oi.DestOutputProfile.N == 4

    page = pdf.pages[0]
    media, trim, bleed = box(page.MediaBox), box(page.TrimBox), box(page.BleedBox)
    assert trim[2] - trim[0] == pytest.approx(210 * MM_TO_PT, abs=0.01)
    assert trim[3] - trim[1] == pytest.approx(297 * MM_TO_PT, abs=0.01)
    assert trim[0] - bleed[0] == pytest.approx(3 * MM_TO_PT, abs=0.01)
    # media box encloses bleed box
    assert media[0] <= bleed[0] and media[2] >= bleed[2]
    assert report.pdf_standard == "PDF/X-1a:2001"


def test_image_is_devicecmyk_and_300dpi():
    pdf_bytes, report = process_image(make_png(), preset="a4", bleed_mm=3)
    pdf = pikepdf.open(io.BytesIO(pdf_bytes))
    im = pdf.pages[0].Resources.XObject.Im0
    assert im.ColorSpace == "/DeviceCMYK"
    assert (int(im.Width), int(im.Height)) == report.pixels
    assert report.pixels == (round(216 / 25.4 * 300), round(303 / 25.4 * 300))


def test_no_rgb_operators_in_content():
    pdf_bytes, _ = process_image(make_png(), preset="a5")
    pdf = pikepdf.open(io.BytesIO(pdf_bytes))
    content = pdf.pages[0].Contents.read_bytes()
    for op in (b" rg", b" RG", b" g\n", b" G\n"):
        assert op not in content
    assert b"1 1 1 1 K" in content  # registration marks


def test_no_marks_option():
    pdf_bytes, report = process_image(make_png(), preset="a5", crop_marks=False)
    pdf = pikepdf.open(io.BytesIO(pdf_bytes))
    assert b"1 1 1 1 K" not in pdf.pages[0].Contents.read_bytes()
    assert report.page_mm == (148 + 6, 210 + 6)


def test_tac_limit():
    # saturated dark colour tends to exceed 300 % in CMYK
    _, report = process_image(make_png(color=(0, 0, 40)), preset="a6")
    assert report.max_tac_percent <= 300.0


def test_transparent_png_flattens_on_white():
    data = make_png(color=(255, 0, 0, 0), mode="RGBA")
    pdf_bytes, _ = process_image(data, preset="a6", crop_marks=False)
    pdf = pikepdf.open(io.BytesIO(pdf_bytes))
    im = pdf.pages[0].Resources.XObject.Im0
    raw = np.frombuffer(im.read_bytes(), dtype=np.uint8).reshape(-1, 4)
    assert raw.max() < 8  # white paper -> no ink


def test_auto_orientation_landscape():
    _, report = process_image(make_png(size=(1000, 600)), preset="a4")
    assert report.orientation == "landscape"
    assert report.trim_mm == (297, 210)


def test_low_res_flag():
    _, report = process_image(make_png(size=(300, 400)), preset="a3")
    assert report.low_resolution is True


def test_bad_inputs():
    with pytest.raises(ProcessingError):
        process_image(b"not an image")
    with pytest.raises(ProcessingError):
        process_image(make_png(), preset="nope")
    with pytest.raises(ProcessingError):
        process_image(make_png(), bleed_mm=50)


def test_api_roundtrip():
    r = client.post(
        "/api/convert",
        files={"file": ("poster.png", make_png(), "image/png")},
        data={"preset": "a5", "bleed_mm": "3", "crop_marks": "true"},
    )
    assert r.status_code == 200
    assert r.headers["content-type"] == "application/pdf"
    assert r.content.startswith(b"%PDF-1.3")
    assert "X-Report" in r.headers


def test_api_rejects_wrong_type_and_garbage():
    r = client.post("/api/convert", files={"file": ("a.txt", b"hi", "text/plain")})
    assert r.status_code == 415
    r = client.post("/api/convert", files={"file": ("a.png", b"junk", "image/png")})
    assert r.status_code == 422
    assert "error" in r.json()


def test_index_and_presets():
    assert client.get("/").status_code == 200
    p = client.get("/api/presets").json()
    assert p["a4"]["w"] == 210
