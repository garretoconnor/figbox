# figbox

Figure/table bounding-box extraction for scientific PDFs (incl. scans).
DocLayout-YOLO (DocStructBench) via ONNX Runtime — no torch at runtime,
~500MB resident, fine on an 8GB M1.

## Setup

Managed with [uv](https://docs.astral.sh/uv/). Two environments: a throwaway one
for the one-time model export (pulls torch), and the lean runtime (no torch).

```bash
# 1. Export the model — one-time, in a throwaway env (torch only needed here).
#    onnxscript is required by torch's ONNX exporter; without it the export
#    fails with "No module named 'onnxscript'".
uv run --no-project --python 3.12 \
  --with doclayout-yolo --with huggingface_hub --with onnx --with onnxslim --with onnxscript \
  python export_model.py          # writes model.onnx (~74MB, gitignored)

# 2. Runtime env (lean — fastapi/onnxruntime/pymupdf, no torch).
uv sync

# 3. (Optional) install the in-repo git hook — formats staged Python with ruff
#    and runs `ruff check` on commit. Skip a run with `git commit --no-verify`.
git config core.hooksPath hooks
```

## Run

```bash
uv run uvicorn server:app --port 8000
```

Env: `MODEL_PATH` (default `model.onnx`), `IMGSZ` (default 1024). Check it's up
with `curl -s localhost:8000/health` → `{"ok": true}`.

## Use

```bash
curl -s -F "file=@paper.pdf" \
  "localhost:8000/detect?dpi=200&classes=figure,table&conf=0.3&pages=1-10" | jq
```

Response per page: `bbox_px` (pixel coords at the requested dpi) and
`bbox_pt` (PDF points, dpi-independent — use these for cropping with
PyMuPDF's `page.get_pixmap(clip=...)`).

Classes: title, plain_text, abandon, figure, figure_caption, table,
table_caption, table_footnote, isolate_formula, formula_caption.
`classes=all` returns everything.

## Knobs

- `dpi`: 200 is a good default; 150 is faster and usually fine for figures.
- `IMGSZ` env var: 1024 default. 800 ≈ halves inference time, figures
  (large objects) barely suffer; small-text classes do.
- `conf`: lower to ~0.15 for faint old scans, then filter downstream.

## Integrating

Calling figbox from another service (e.g. to crop figures into a notes repo)?
See [INTEGRATION.md](INTEGRATION.md): the `/detect` contract, a real response,
and the render/crop/caption recipe (figbox returns boxes only — the caller crops
and OCRs captions).
