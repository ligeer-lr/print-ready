"""Print-ready PDF/X-1a generator.

Takes an RGB image (e.g. from Midjourney / DALL·E) and produces a PDF/X-1a:2001
file with: 300 DPI raster, bleed, DeviceCMYK image (FOGRA39 conversion),
TAC limit, crop marks, TrimBox/BleedBox and an embedded OutputIntent.
"""
from __future__ import annotations

import io
import zlib
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pikepdf
from PIL import Image, ImageCms, ImageOps
from pikepdf import Array, Dictionary, Name, Pdf, String

MM_TO_PT = 72.0 / 25.4
ICC_PATH = Path(__file__).parent / "icc" / "FOGRA39L_coated.icc"
OUTPUT_CONDITION = "FOGRA39 (ISO 12647-2:2004)"
OUTPUT_CONDITION_ID = "FOGRA39"

MAX_SOURCE_PIXELS = 100_000_000  # decompression-bomb guard
MAX_OUTPUT_PIXELS = 120_000_000
TAC_LIMIT = 300  # percent

# trim size in mm (width, height) in portrait
PRESETS: dict[str, dict] = {
    "a6": {"w": 105, "h": 148, "ar": "A6", "en": "A6 (postcard)"},
    "a5": {"w": 148, "h": 210, "ar": "A5 (فلاير)", "en": "A5 (flyer)"},
    "a4": {"w": 210, "h": 297, "ar": "A4", "en": "A4"},
    "a3": {"w": 297, "h": 420, "ar": "A3 (بوستر)", "en": "A3 (poster)"},
    "a2": {"w": 420, "h": 594, "ar": "A2 (بوستر كبير)", "en": "A2 (large poster)"},
    "business_card": {"w": 90, "h": 50, "ar": "بطاقة أعمال", "en": "Business card"},
}


class ProcessingError(ValueError):
    """Raised for user-fixable input problems."""


@dataclass
class Report:
    preset: str
    orientation: str
    trim_mm: tuple[float, float]
    bleed_mm: float
    dpi: int
    pixels: tuple[int, int]
    source_pixels: tuple[int, int]
    source_effective_dpi: int
    upscale_factor: float
    low_resolution: bool
    max_tac_percent: float
    crop_marks: bool
    pdf_standard: str
    color: str
    page_mm: tuple[float, float]

    def as_dict(self) -> dict:
        return asdict(self)


def _load_image(data: bytes) -> tuple[Image.Image, bytes | None]:
    Image.MAX_IMAGE_PIXELS = MAX_SOURCE_PIXELS
    try:
        img = Image.open(io.BytesIO(data))
        img.load()
    except Image.DecompressionBombError as exc:
        raise ProcessingError("الصورة كبيرة جداً / Image is too large") from exc
    except Exception as exc:  # noqa: BLE001
        raise ProcessingError("تعذّر قراءة الصورة / Could not read the image") from exc

    icc = img.info.get("icc_profile")
    img = ImageOps.exif_transpose(img)

    # flatten transparency on white (PDF/X-1a forbids transparency)
    if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
        rgba = img.convert("RGBA")
        bg = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        img = Image.alpha_composite(bg, rgba).convert("RGB")
    elif img.mode == "CMYK":
        pass
    else:
        img = img.convert("RGB")
    return img, icc


def _to_srgb(img: Image.Image, icc: bytes | None) -> Image.Image:
    if img.mode != "RGB":
        return img
    if not icc:
        return img
    try:
        src = ImageCms.ImageCmsProfile(io.BytesIO(icc))
        dst = ImageCms.createProfile("sRGB")
        return ImageCms.profileToProfile(img, src, dst, outputMode="RGB")
    except Exception:  # noqa: BLE001  broken profile -> assume sRGB
        return img


def _cover_resize(img: Image.Image, size: tuple[int, int]) -> Image.Image:
    """Scale to fully cover `size`, centre-crop the overflow."""
    tw, th = size
    sw, sh = img.size
    scale = max(tw / sw, th / sh)
    nw, nh = max(tw, round(sw * scale)), max(th, round(sh * scale))
    resample = Image.LANCZOS
    resized = img.resize((nw, nh), resample)
    left = (nw - tw) // 2
    top = (nh - th) // 2
    return resized.crop((left, top, left + tw, top + th))


def _rgb_to_cmyk(img: Image.Image) -> Image.Image:
    src = ImageCms.createProfile("sRGB")
    dst = ImageCms.getOpenProfile(str(ICC_PATH))
    transform = ImageCms.buildTransform(
        src,
        dst,
        "RGB",
        "CMYK",
        renderingIntent=ImageCms.Intent.RELATIVE_COLORIMETRIC,
        flags=ImageCms.Flags.BLACKPOINTCOMPENSATION,
    )
    return ImageCms.applyTransform(img, transform)


def _limit_tac(cmyk: Image.Image, limit: int = TAC_LIMIT) -> tuple[Image.Image, float]:
    """Scale CMY down where total ink exceeds `limit` percent. Returns image, max TAC."""
    arr = np.asarray(cmyk, dtype=np.uint16)  # PIL CMYK: 0..255 = 0..100 %
    total = arr.sum(axis=2).astype(np.float32) * (100.0 / 255.0)
    max_before = float(total.max()) if total.size else 0.0
    if max_before <= limit:
        return cmyk, max_before
    over = total > limit
    k = arr[..., 3].astype(np.float32) * (100.0 / 255.0)
    cmy = total - k
    allowed_cmy = np.maximum(limit - k, 0.0)
    scale = np.ones_like(total)
    safe = np.where(cmy > 0, cmy, 1.0)
    scale[over] = (allowed_cmy[over] / safe[over]).clip(0, 1)
    out = arr.astype(np.float32)
    for ch in range(3):
        out[..., ch] *= scale
    out = np.clip(np.floor(out), 0, 255).astype(np.uint8)
    limited = Image.fromarray(out, mode="CMYK")
    new_total = (out.astype(np.uint16).sum(axis=2).astype(np.float32) * (100.0 / 255.0)).max()
    return limited, float(new_total)


def _crop_mark_stream(
    page_w: float, page_h: float, trim_x: float, trim_y: float, trim_w: float, trim_h: float,
    offset: float, length: float,
) -> bytes:
    """Registration-black (1 1 1 1) crop marks at the four trim corners, 0.25 pt."""
    x0, x1 = trim_x, trim_x + trim_w
    y0, y1 = trim_y, trim_y + trim_h
    lines: list[tuple[float, float, float, float]] = []
    for x in (x0, x1):
        lines.append((x, y0 - offset, x, y0 - offset - length))  # bottom
        lines.append((x, y1 + offset, x, y1 + offset + length))  # top
    for y in (y0, y1):
        lines.append((x0 - offset, y, x0 - offset - length, y))  # left
        lines.append((x1 + offset, y, x1 + offset + length, y))  # right
    ops = ["q", "0.25 w", "1 1 1 1 K"]
    for ax, ay, bx, by in lines:
        ops.append(f"{ax:.3f} {ay:.3f} m {bx:.3f} {by:.3f} l S")
    ops.append("Q")
    return "\n".join(ops).encode("ascii")


def _pdf_date(dt: datetime) -> str:
    return dt.strftime("D:%Y%m%d%H%M%S+00'00'")


def build_pdf(
    cmyk: Image.Image,
    *,
    trim_w_mm: float,
    trim_h_mm: float,
    bleed_mm: float,
    crop_marks: bool,
    title: str,
) -> bytes:
    slug_mm = max(bleed_mm, 10.0) if crop_marks else bleed_mm
    page_w_mm = trim_w_mm + 2 * slug_mm
    page_h_mm = trim_h_mm + 2 * slug_mm

    page_w, page_h = page_w_mm * MM_TO_PT, page_h_mm * MM_TO_PT
    trim_x = trim_y = slug_mm * MM_TO_PT
    trim_w, trim_h = trim_w_mm * MM_TO_PT, trim_h_mm * MM_TO_PT
    bleed = bleed_mm * MM_TO_PT
    img_w_pt, img_h_pt = trim_w + 2 * bleed, trim_h + 2 * bleed

    pdf = Pdf.new()

    # --- image (DeviceCMYK, Flate) ---
    raw = cmyk.tobytes()
    image = pikepdf.Stream(pdf, zlib.compress(raw, 6))
    image.Type = Name.XObject
    image.Subtype = Name.Image
    image.Width, image.Height = cmyk.size
    image.ColorSpace = Name.DeviceCMYK
    image.BitsPerComponent = 8
    image.Filter = Name.FlateDecode
    del raw

    # --- content ---
    content_ops = [
        "q",
        f"{img_w_pt:.4f} 0 0 {img_h_pt:.4f} {trim_x - bleed:.4f} {trim_y - bleed:.4f} cm",
        "/Im0 Do",
        "Q",
    ]
    content = "\n".join(content_ops).encode("ascii")
    if crop_marks:
        offset = bleed_mm * MM_TO_PT if bleed_mm >= 3 else 3 * MM_TO_PT
        length = min(5 * MM_TO_PT, slug_mm * MM_TO_PT - offset - 0.5)
        if length > 1:
            content += b"\n" + _crop_mark_stream(
                page_w, page_h, trim_x, trim_y, trim_w, trim_h, offset, length
            )

    page = pdf.make_indirect(
        Dictionary(
            Type=Name.Page,
            MediaBox=Array([0, 0, page_w, page_h]),
            TrimBox=Array([trim_x, trim_y, trim_x + trim_w, trim_y + trim_h]),
            BleedBox=Array([trim_x - bleed, trim_y - bleed, trim_x + trim_w + bleed, trim_y + trim_h + bleed]),
            Resources=Dictionary(XObject=Dictionary(Im0=image)),
            Contents=pikepdf.Stream(pdf, content),
        )
    )
    pdf.pages.append(pikepdf.Page(page))

    # --- output intent ---
    icc_bytes = ICC_PATH.read_bytes()
    icc_stream = pikepdf.Stream(pdf, icc_bytes)
    icc_stream.N = 4
    icc_stream.Alternate = Name.DeviceCMYK
    output_intent = pdf.make_indirect(
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
    pdf.Root.OutputIntents = Array([output_intent])

    # --- Info dictionary required by PDF/X-1a:2001 ---
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
    pdf.save(out, min_version="1.3", force_version="1.3", object_stream_mode=pikepdf.ObjectStreamMode.disable,
             deterministic_id=False)
    return out.getvalue()


def process_image(
    data: bytes,
    *,
    preset: str = "a4",
    bleed_mm: float = 3.0,
    dpi: int = 300,
    crop_marks: bool = True,
    landscape: bool | None = None,
    filename: str = "artwork",
) -> tuple[bytes, Report]:
    if preset not in PRESETS:
        raise ProcessingError(f"Unknown size: {preset}")
    if not (0 <= bleed_mm <= 10):
        raise ProcessingError("Bleed must be between 0 and 10 mm")
    if not (150 <= dpi <= 600):
        raise ProcessingError("DPI must be between 150 and 600")

    img, icc = _load_image(data)
    src_size = img.size

    tw, th = PRESETS[preset]["w"], PRESETS[preset]["h"]
    if landscape is None:  # auto-match orientation to the artwork
        landscape = src_size[0] > src_size[1]
    if landscape and tw < th:
        tw, th = th, tw
    elif not landscape and tw > th:
        tw, th = th, tw

    px_w = round((tw + 2 * bleed_mm) / 25.4 * dpi)
    px_h = round((th + 2 * bleed_mm) / 25.4 * dpi)
    if px_w * px_h > MAX_OUTPUT_PIXELS:
        raise ProcessingError("الحجم المطلوب كبير جداً / Requested size is too large")

    scale = max(px_w / src_size[0], px_h / src_size[1])
    eff_dpi = round(min(src_size[0] / ((tw + 2 * bleed_mm) / 25.4), src_size[1] / ((th + 2 * bleed_mm) / 25.4)))

    img = _to_srgb(img, icc) if img.mode == "RGB" else img
    img = _cover_resize(img, (px_w, px_h))
    cmyk = img if img.mode == "CMYK" else _rgb_to_cmyk(img)
    cmyk, max_tac = _limit_tac(cmyk)

    pdf_bytes = build_pdf(
        cmyk,
        trim_w_mm=tw,
        trim_h_mm=th,
        bleed_mm=bleed_mm,
        crop_marks=crop_marks,
        title=Path(filename).stem[:80] or "artwork",
    )
    slug = max(bleed_mm, 10.0) if crop_marks else bleed_mm
    report = Report(
        preset=preset,
        orientation="landscape" if tw > th else "portrait",
        trim_mm=(tw, th),
        bleed_mm=bleed_mm,
        dpi=dpi,
        pixels=(px_w, px_h),
        source_pixels=src_size,
        source_effective_dpi=eff_dpi,
        upscale_factor=round(scale, 2),
        low_resolution=eff_dpi < 150,
        max_tac_percent=round(max_tac, 1),
        crop_marks=crop_marks,
        pdf_standard="PDF/X-1a:2001",
        color=OUTPUT_CONDITION,
        page_mm=(tw + 2 * slug, th + 2 * slug),
    )
    return pdf_bytes, report
