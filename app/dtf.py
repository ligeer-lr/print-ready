"""DTF mode: background removal, clean white plate, PNG + PDF/X-1a with a White spot plate.

Output bundle (zip):
  artwork.png        RGBA, 300 DPI, hard (non half-transparent) edges, tight crop
  white-plate.png    black on white; black = where white ink goes (choked a little)
  artwork-dtf.pdf    PDF/X-1a, CMYK artwork + spot colour "White" underbase
"""
from __future__ import annotations

import io
import os
import zipfile
import zlib
from dataclasses import dataclass, asdict
from datetime import datetime, timezone

import numpy as np
import pikepdf
from PIL import Image, ImageFilter, ImageOps
from pikepdf import Array, Dictionary, Name, Pdf, String

from .processor import (
    ICC_PATH,
    MM_TO_PT,
    OUTPUT_CONDITION,
    OUTPUT_CONDITION_ID,
    ProcessingError,
    _limit_tac,
    _load_image,
    _pdf_date,
    _rgb_to_cmyk,
    _to_srgb,
)

REMBG_MODEL = os.environ.get("REMBG_MODEL", "u2netp")
MARGIN_MM = 5.0
MAX_PIXELS = 100_000_000
_session = None


@dataclass
class DtfReport:
    width_cm: float
    height_cm: float
    dpi: int
    pixels: tuple[int, int]
    source_pixels: tuple[int, int]
    source_effective_dpi: int
    background_removed: bool
    choke_mm: float
    white_coverage_percent: float
    max_tac_percent: float
    low_resolution: bool
    pdf_standard: str = "PDF/X-1a:2001 + White spot"

    def as_dict(self) -> dict:
        return asdict(self)


def _rembg_session():
    global _session
    if _session is None:
        from rembg import new_session  # heavy import, load lazily

        _session = new_session(REMBG_MODEL)
    return _session


def _remove_bg(img: Image.Image) -> Image.Image:
    from rembg import remove

    return remove(img.convert("RGB"), session=_rembg_session()).convert("RGBA")


def _border_colour(arr: np.ndarray) -> tuple[np.ndarray, float]:
    """Median colour of the image border and how uniform it is (mean distance)."""
    h, w = arr.shape[:2]
    t = max(2, round(min(h, w) * 0.02))
    ring = np.concatenate(
        [arr[:t].reshape(-1, 3), arr[-t:].reshape(-1, 3), arr[:, :t].reshape(-1, 3), arr[:, -t:].reshape(-1, 3)]
    ).astype(np.float32)
    med = np.median(ring, axis=0)
    spread = float(np.linalg.norm(ring - med, axis=1).mean())
    return med, spread


def _color_key(img: Image.Image, bg: np.ndarray, tol: float = 30.0) -> Image.Image:
    """Remove the flat background by flood-filling from the edges.

    Only background pixels connected to the border go, so same-coloured details
    inside the artwork stay. Keeps lettering that AI cut-out models tend to eat.
    """
    from scipy import ndimage

    arr = np.asarray(img.convert("RGB"), dtype=np.float32)
    near = np.linalg.norm(arr - bg, axis=2) < tol
    labels, _ = ndimage.label(near)
    border = np.unique(np.concatenate([labels[0], labels[-1], labels[:, 0], labels[:, -1]]))
    border = border[border != 0]
    outside = np.isin(labels, border)
    alpha = Image.fromarray(np.where(outside, 0, 255).astype(np.uint8), "L")
    alpha = alpha.filter(ImageFilter.MinFilter(3))  # strip the bg-coloured halo
    rgba = img.convert("RGBA")
    rgba.putalpha(alpha)
    return rgba


def _binary_mask(alpha: Image.Image, dpi: int, choke_mm: float) -> tuple[Image.Image, Image.Image]:
    """Return (color_mask, white_mask) as 'L' images with values 0/255.

    - threshold removes half-transparent edges (they print as grey fuzz)
    - opening removes isolated specks
    - white mask is choked so white ink never shows outside the colour
    """
    choke_px = max(0, round(choke_mm / 25.4 * dpi))
    pad = choke_px + 2  # filters repeat edge pixels, so pad with empty space first
    mask = alpha.point(lambda v: 255 if v >= 128 else 0, "L")
    mask = ImageOps.expand(mask, border=pad, fill=0)
    mask = mask.filter(ImageFilter.MinFilter(3)).filter(ImageFilter.MaxFilter(3))  # opening
    white = mask
    for _ in range(choke_px):
        white = white.filter(ImageFilter.MinFilter(3))
    box = (pad, pad, pad + alpha.width, pad + alpha.height)
    return mask.crop(box), white.crop(box)


def _pack_mask(mask: Image.Image) -> bytes:
    """1-bit stencil: 0 = paint, 1 = masked out (PDF default Decode [0 1])."""
    arr = np.asarray(mask) < 128  # True where NOT painted
    return np.packbits(arr, axis=1).tobytes()


def _build_pdf(
    cmyk: Image.Image, color_mask: Image.Image, white_mask: Image.Image, *, w_mm: float, h_mm: float, title: str
) -> bytes:
    page_w = (w_mm + 2 * MARGIN_MM) * MM_TO_PT
    page_h = (h_mm + 2 * MARGIN_MM) * MM_TO_PT
    m = MARGIN_MM * MM_TO_PT
    art_w, art_h = w_mm * MM_TO_PT, h_mm * MM_TO_PT
    px_w, px_h = cmyk.size

    pdf = Pdf.new()

    # colour image with explicit stencil mask (hard edge, no transparency)
    stencil = pikepdf.Stream(pdf, zlib.compress(_pack_mask(color_mask), 6))
    stencil.Type = Name.XObject
    stencil.Subtype = Name.Image
    stencil.Width, stencil.Height = px_w, px_h
    stencil.ImageMask = True
    stencil.BitsPerComponent = 1
    stencil.Filter = Name.FlateDecode

    image = pikepdf.Stream(pdf, zlib.compress(cmyk.tobytes(), 6))
    image.Type = Name.XObject
    image.Subtype = Name.Image
    image.Width, image.Height = px_w, px_h
    image.ColorSpace = Name.DeviceCMYK
    image.BitsPerComponent = 8
    image.Filter = Name.FlateDecode
    image.Mask = stencil

    # white plate: same-size stencil painted with the White separation
    white = pikepdf.Stream(pdf, zlib.compress(_pack_mask(white_mask), 6))
    white.Type = Name.XObject
    white.Subtype = Name.Image
    white.Width, white.Height = px_w, px_h
    white.ImageMask = True
    white.BitsPerComponent = 1
    white.Filter = Name.FlateDecode

    tint = pdf.make_indirect(
        Dictionary(
            FunctionType=2,
            Domain=Array([0, 1]),
            C0=Array([0, 0, 0, 0]),
            C1=Array([0.0, 0.0, 0.0, 0.15]),
            N=1,
        )
    )
    separation = Array([Name.Separation, Name("/White"), Name.DeviceCMYK, tint])

    overprint = Dictionary(Type=Name.ExtGState, OP=True, op=True, OPM=1)

    ops = [
        "q",
        "/CS0 cs 1 scn",
        f"{art_w:.4f} 0 0 {art_h:.4f} {m:.4f} {m:.4f} cm",
        "/Wh Do",
        "Q",
        "q",
        "/GS0 gs",
        f"{art_w:.4f} 0 0 {art_h:.4f} {m:.4f} {m:.4f} cm",
        "/Im0 Do",
        "Q",
    ]
    page = pdf.make_indirect(
        Dictionary(
            Type=Name.Page,
            MediaBox=Array([0, 0, page_w, page_h]),
            TrimBox=Array([m, m, m + art_w, m + art_h]),
            BleedBox=Array([m, m, m + art_w, m + art_h]),
            Resources=Dictionary(
                XObject=Dictionary(Im0=image, Wh=white),
                ColorSpace=Dictionary(CS0=separation),
                ExtGState=Dictionary(GS0=overprint),
            ),
            Contents=pikepdf.Stream(pdf, "\n".join(ops).encode("ascii")),
        )
    )
    pdf.pages.append(pikepdf.Page(page))

    icc_stream = pikepdf.Stream(pdf, ICC_PATH.read_bytes())
    icc_stream.N = 4
    icc_stream.Alternate = Name.DeviceCMYK
    pdf.Root.OutputIntents = Array(
        [
            pdf.make_indirect(
                Dictionary(
                    Type=Name.OutputIntent,
                    S=Name("/GTS_PDFX"),
                    OutputConditionIdentifier=String(OUTPUT_CONDITION_ID),
                    OutputCondition=String(OUTPUT_CONDITION),
                    RegistryName=String("http://www.color.org"),
                    Info=String(OUTPUT_CONDITION),
                    DestOutputProfile=icc_stream,
                )
            )
        ]
    )
    now = _pdf_date(datetime.now(timezone.utc))
    pdf.docinfo = pdf.make_indirect(
        Dictionary(
            Title=String(title),
            Producer=String("print-ready"),
            Creator=String("print-ready"),
            CreationDate=String(now),
            ModDate=String(now),
            Trapped=Name("/False"),
            GTS_PDFXVersion=String("PDF/X-1a:2001"),
            GTS_PDFXConformance=String("PDF/X-1a:2001"),
        )
    )
    out = io.BytesIO()
    pdf.save(out, min_version="1.3", force_version="1.3", object_stream_mode=pikepdf.ObjectStreamMode.disable)
    return out.getvalue()


def process_dtf(
    data: bytes,
    *,
    width_cm: float = 25.0,
    dpi: int = 300,
    remove_background: bool = True,
    bg_method: str = "auto",
    choke_mm: float = 0.2,
    filename: str = "artwork",
) -> tuple[bytes, DtfReport]:
    if not (2 <= width_cm <= 60):
        raise ProcessingError("العرض بين 2 و 60 سم / Width must be 2-60 cm")
    if not (150 <= dpi <= 600):
        raise ProcessingError("DPI must be between 150 and 600")
    if bg_method not in {"auto", "color", "ai"}:
        raise ProcessingError("Invalid background method")
    if not (0 <= choke_mm <= 1):
        raise ProcessingError("Choke must be between 0 and 1 mm")

    # keep the original alpha if the file already has transparency
    Image.MAX_IMAGE_PIXELS = 100_000_000
    try:
        raw = Image.open(io.BytesIO(data))
        has_alpha = raw.mode in ("RGBA", "LA", "PA") or (raw.mode == "P" and "transparency" in raw.info)
    except Exception as exc:  # noqa: BLE001
        raise ProcessingError("تعذّر قراءة الصورة / Could not read the image") from exc

    if has_alpha:
        rgba = ImageOps.exif_transpose(raw).convert("RGBA")
        removed = False
    else:
        img, icc = _load_image(data)
        img = _to_srgb(img.convert("RGB"), icc)
        if remove_background:
            bg, spread = _border_colour(np.asarray(img))
            use_key = bg_method == "color" or (bg_method == "auto" and spread < 12)
            rgba = _color_key(img, bg) if use_key else _remove_bg(img)
            removed = True
        else:
            rgba = img.convert("RGBA")
            removed = False

    src_size = rgba.size
    alpha = rgba.getchannel("A")
    bbox = alpha.point(lambda v: 255 if v >= 128 else 0).getbbox()
    if bbox is None:
        raise ProcessingError("ما لقيت تصميم في الصورة / No artwork found after background removal")
    rgba = rgba.crop(bbox)

    cw, ch = rgba.size
    out_w = round(width_cm / 2.54 * dpi)
    out_h = round(out_w * ch / cw)
    if out_w * out_h > MAX_PIXELS:
        raise ProcessingError("الحجم المطلوب كبير جداً / Requested size is too large")
    height_cm = out_h / dpi * 2.54
    eff_dpi = round(cw / (width_cm / 2.54))

    # premultiply-safe resize: resize colour on opaque-filled edges to avoid dark fringes
    rgb = rgba.convert("RGB")
    a = rgba.getchannel("A")
    solid = Image.composite(rgb, rgb.filter(ImageFilter.MaxFilter(5)), a.point(lambda v: 255 if v > 0 else 0))
    rgb_big = solid.resize((out_w, out_h), Image.LANCZOS)
    a_big = a.resize((out_w, out_h), Image.LANCZOS)

    color_mask, white_mask = _binary_mask(a_big, dpi, choke_mm)
    coverage = float(np.asarray(white_mask).mean() / 255.0 * 100.0)

    art_png = rgb_big.copy()
    art_png.putalpha(color_mask)
    cmyk = _rgb_to_cmyk(rgb_big)
    cmyk, max_tac = _limit_tac(cmyk)

    title = (os.path.splitext(os.path.basename(filename))[0] or "artwork")[:80]
    pdf_bytes = _build_pdf(cmyk, color_mask, white_mask, w_mm=width_cm * 10, h_mm=height_cm * 10, title=title)

    buf = io.BytesIO()
    art_png.save(buf, "PNG", dpi=(dpi, dpi))
    art_png_bytes = buf.getvalue()
    plate = ImageOps.invert(white_mask).convert("L")  # black = ink
    buf = io.BytesIO()
    plate.save(buf, "PNG", dpi=(dpi, dpi))
    plate_bytes = buf.getvalue()

    zbuf = io.BytesIO()
    with zipfile.ZipFile(zbuf, "w", zipfile.ZIP_STORED) as z:
        z.writestr("artwork.png", art_png_bytes)
        z.writestr("white-plate.png", plate_bytes)
        z.writestr("artwork-dtf.pdf", pdf_bytes)

    report = DtfReport(
        width_cm=round(width_cm, 2),
        height_cm=round(height_cm, 2),
        dpi=dpi,
        pixels=(out_w, out_h),
        source_pixels=src_size,
        source_effective_dpi=eff_dpi,
        background_removed=removed,
        choke_mm=choke_mm,
        white_coverage_percent=round(coverage, 1),
        max_tac_percent=round(max_tac, 1),
        low_resolution=eff_dpi < 150,
    )
    return zbuf.getvalue(), report
