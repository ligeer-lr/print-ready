from __future__ import annotations

import json
from pathlib import Path
from urllib.parse import quote

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool

from .dtf import preview_cutout, process_dtf
from .processor import PRESETS, ProcessingError, process_image

MAX_UPLOAD_BYTES = 25 * 1024 * 1024
ALLOWED_TYPES = {"image/png", "image/jpeg", "image/webp"}
STATIC_DIR = Path(__file__).parent / "static"

app = FastAPI(title="print-ready", version="0.1.0")


@app.get("/api/presets")
def presets() -> dict:
    return {k: {"w": v["w"], "h": v["h"], "ar": v["ar"], "en": v["en"]} for k, v in PRESETS.items()}


@app.get("/api/health")
def health() -> dict:
    return {"ok": True}


@app.post("/api/convert")
async def convert(
    file: UploadFile = File(...),
    preset: str = Form("a4"),
    bleed_mm: float = Form(3.0),
    dpi: int = Form(300),
    crop_marks: bool = Form(True),
    orientation: str = Form("auto"),
) -> Response:
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(415, "PNG / JPG / WebP only")
    data = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "File is larger than 25 MB")
    if orientation not in {"auto", "portrait", "landscape"}:
        raise HTTPException(422, "Invalid orientation")
    landscape = None if orientation == "auto" else orientation == "landscape"

    try:
        pdf, report = await run_in_threadpool(
            process_image,
            data,
            preset=preset,
            bleed_mm=bleed_mm,
            dpi=dpi,
            crop_marks=crop_marks,
            landscape=landscape,
            filename=file.filename or "artwork",
        )
    except ProcessingError as exc:
        raise HTTPException(422, str(exc)) from exc

    stem = Path(file.filename or "artwork").stem[:60] or "artwork"
    name = f"{stem}-{preset}-print-ready.pdf"
    return Response(
        content=pdf,
        media_type="application/pdf",
        headers={
            "Content-Disposition": f"attachment; filename*=UTF-8''{quote(name)}",
            "X-Report": quote(json.dumps(report.as_dict())),
            "Access-Control-Expose-Headers": "X-Report, Content-Disposition",
        },
    )


@app.post("/api/dtf/preview")
async def dtf_preview(
    file: UploadFile = File(...),
    bg_method: str = Form("auto"),
    picks: str = Form(""),
    tolerance: float = Form(30.0),
    scope: str = Form("contiguous"),
) -> Response:
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(415, "PNG / JPG / WebP only")
    data = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "File is larger than 25 MB")
    try:
        png = await run_in_threadpool(
            preview_cutout, data, bg_method=bg_method, picks=picks, tolerance=tolerance, scope=scope
        )
    except ProcessingError as exc:
        raise HTTPException(422, str(exc)) from exc
    return Response(content=png, media_type="image/png")


@app.post("/api/dtf")
async def dtf(
    file: UploadFile = File(...),
    width_cm: float = Form(25.0),
    remove_background: bool = Form(True),
    bg_method: str = Form("auto"),
    choke_mm: float = Form(0.2),
    picks: str = Form(""),
    tolerance: float = Form(30.0),
    scope: str = Form("contiguous"),
) -> Response:
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(415, "PNG / JPG / WebP only")
    data = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "File is larger than 25 MB")
    try:
        bundle, report = await run_in_threadpool(
            process_dtf,
            data,
            width_cm=width_cm,
            remove_background=remove_background,
            bg_method=bg_method,
            choke_mm=choke_mm,
            picks=picks,
            tolerance=tolerance,
            scope=scope,
            filename=file.filename or "artwork",
        )
    except ProcessingError as exc:
        raise HTTPException(422, str(exc)) from exc

    stem = Path(file.filename or "artwork").stem[:60] or "artwork"
    return Response(
        content=bundle,
        media_type="application/zip",
        headers={
            "Content-Disposition": f"attachment; filename*=UTF-8''{quote(stem + '-dtf.zip')}",
            "X-Report": quote(json.dumps(report.as_dict())),
            "Access-Control-Expose-Headers": "X-Report, Content-Disposition",
        },
    )


@app.exception_handler(HTTPException)
async def http_error(_, exc: HTTPException) -> JSONResponse:
    return JSONResponse({"error": exc.detail}, status_code=exc.status_code)


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
