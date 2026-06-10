# figbox

Figure/table bounding-box extraction for scientific PDFs (incl. scans).
DocLayout-YOLO (DocStructBench) via ONNX Runtime — no torch at runtime,
~500MB resident, fine on an 8GB M1.

## Setup

```bash
# 1. Export the model (throwaway venv — torch is only needed here)
python -m venv /tmp/export-venv && source /tmp/export-venv/bin/activate
pip install doclayout-yolo huggingface_hub onnx onnxslim
python export_model.py          # writes model.onnx (~80MB)
deactivate && rm -rf /tmp/export-venv

# 2. Runtime deps (lean)
python -m venv .venv && source .venv/bin/activate
pip install fastapi uvicorn[standard] onnxruntime pymupdf pillow numpy
```

## Run

```bash
uvicorn server:app --port 8000
```

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
