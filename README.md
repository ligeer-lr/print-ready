# Print Ready

حوّل أي صورة (Midjourney / DALL·E / أي تصميم) إلى ملف طباعة **PDF/X-1a:2001** جاهز للمطبعة.
Turn any image into a press-ready **PDF/X-1a:2001** file.

## What it does

- Resamples to the requested size at 300 DPI
- Adds bleed (default 3 mm) and crop marks
- Converts sRGB → CMYK with the FOGRA39 profile, limits total ink to 300 %
- Writes TrimBox / BleedBox, embeds the output intent, no RGB or transparency
- Arabic / English interface

Sizes: A6, A5, A4, A3, A2, business card (portrait / landscape / auto).

## Run locally

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload
# open http://127.0.0.1:8000
```

## Run from GitHub (no install)

1. Open the repo on GitHub, click **Code** → **Codespaces** → **Create codespace on main**.
2. Wait for it to start. The app launches by itself and a browser tab opens on port 8000.
   If it does not, open the **Ports** tab and click the link next to 8000.

## Tests

```bash
pip install -r requirements-dev.txt
pytest
```

## Deploy

A `Dockerfile` is included. It runs on Railway, Render, Fly.io or any VPS:

```bash
docker build -t print-ready .
docker run -p 8000:8000 print-ready
```

Render.com: **New → Blueprint**, pick this repo. `render.yaml` does the rest.
The free plan has little memory, so large sizes (A3 / A2) may fail there; use a paid plan for those.

## Layout

```
app/processor.py   image → CMYK → PDF/X-1a
app/main.py        FastAPI: POST /api/convert, GET /api/presets
app/static/        single-page UI
app/icc/           FOGRA39L coated ICC profile
tests/             pytest suite
```

## Roadmap

1. AI upscaling (Real-ESRGAN) instead of plain resampling
2. DTF mode: white underbase plate + background removal
3. Accounts, credits and subscriptions (Moyasar / Stripe)
4. More sizes and a custom-size field

## Notes

The FOGRA39L coated profile comes from the TeX Live `colorprofiles` package.
Check its licence before selling the service commercially, or swap in a profile you hold the rights to.
