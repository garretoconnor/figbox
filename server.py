"""FastAPI service: POST a PDF, get figure/table bounding boxes back.

Run:  uvicorn server:app --port 8000
Env:  MODEL_PATH (default: model.onnx), IMGSZ (default: 1024)

Example:
  curl -s -F "file=@paper.pdf" \
    "localhost:8000/detect?dpi=200&classes=figure,table&conf=0.3" | jq
"""

import io
import os

import fitz  # PyMuPDF
from fastapi import FastAPI, HTTPException, Query, UploadFile
from PIL import Image

from detector import CLASSES, LayoutDetector

app = FastAPI(title="figbox")
detector: LayoutDetector | None = None


@app.on_event("startup")
def load_model():
    global detector
    model_path = os.environ.get("MODEL_PATH", "model.onnx")
    imgsz = int(os.environ.get("IMGSZ", "1024"))
    detector = LayoutDetector(model_path, imgsz=imgsz)


def parse_pages(spec: str | None, n_pages: int) -> list[int]:
    """'1-3,7' -> [0, 1, 2, 6] (0-indexed). None -> all pages."""
    if not spec:
        return list(range(n_pages))
    out: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-", 1)
            out.update(range(int(a) - 1, int(b)))
        else:
            out.add(int(part) - 1)
    return sorted(i for i in out if 0 <= i < n_pages)


@app.post("/detect")
async def detect(
    file: UploadFile,
    dpi: int = Query(200, ge=72, le=400),
    conf: float = Query(0.25, ge=0.0, le=1.0),
    classes: str = Query("figure", description="comma-separated, or 'all'"),
    pages: str | None = Query(None, description="1-indexed, e.g. '1-3,7'"),
):
    if detector is None:
        raise HTTPException(503, "model not loaded")

    wanted = None if classes == "all" else {c.strip() for c in classes.split(",")}
    if wanted is not None and not wanted <= set(CLASSES.values()):
        raise HTTPException(400, f"unknown classes; valid: {sorted(CLASSES.values())}")

    data = await file.read()
    try:
        doc = fitz.open(stream=data, filetype="pdf")
    except Exception as e:
        raise HTTPException(400, f"could not open PDF: {e}")

    scale = dpi / 72.0
    results = []
    for i in parse_pages(pages, doc.page_count):
        page = doc[i]
        pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale), colorspace=fitz.csRGB)
        img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)

        dets = detector.detect(img, conf=conf, classes=wanted)
        for d in dets:
            # PDF points (72/inch) so coords are usable independent of dpi
            d["bbox_pt"] = [round(v / scale, 2) for v in d["bbox_px"]]

        results.append(
            {
                "page": i + 1,
                "size_px": [pix.width, pix.height],
                "size_pt": [round(page.rect.width, 2), round(page.rect.height, 2)],
                "detections": dets,
            }
        )
    doc.close()
    return {"dpi": dpi, "pages": results}


@app.get("/health")
def health():
    return {"ok": detector is not None}
