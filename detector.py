"""ONNX inference for DocLayout-YOLO (DocStructBench weights).

The DocStructBench checkpoint is YOLOv10-based, so the ONNX export is
end-to-end (NMS baked in) and emits (1, N, 6) rows of
[x1, y1, x2, y2, score, class] in letterboxed-input coordinates.
No anchor decoding or NMS needed here.
"""

import numpy as np
import onnxruntime as ort
from PIL import Image

CLASSES = {
    0: "title",
    1: "plain_text",
    2: "abandon",          # headers/footers/page numbers
    3: "figure",
    4: "figure_caption",
    5: "table",
    6: "table_caption",
    7: "table_footnote",
    8: "isolate_formula",
    9: "formula_caption",
}


class LayoutDetector:
    def __init__(self, model_path: str, imgsz: int = 1024, num_threads: int = 4):
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = num_threads
        self.session = ort.InferenceSession(
            model_path, sess_options=opts, providers=["CPUExecutionProvider"]
        )
        self.input_name = self.session.get_inputs()[0].name
        self.imgsz = imgsz

    def _letterbox(self, img: Image.Image):
        """Resize keeping aspect ratio, pad to imgsz x imgsz."""
        w, h = img.size
        r = self.imgsz / max(w, h)
        nw, nh = round(w * r), round(h * r)
        resized = img.resize((nw, nh), Image.BILINEAR)
        canvas = Image.new("RGB", (self.imgsz, self.imgsz), (114, 114, 114))
        left = (self.imgsz - nw) // 2
        top = (self.imgsz - nh) // 2
        canvas.paste(resized, (left, top))
        return canvas, r, left, top

    def detect(self, img: Image.Image, conf: float = 0.25, classes: set | None = None):
        """Run detection on a PIL image.

        Returns a list of dicts with class, confidence, and bbox_px
        (pixel coords in the original image's coordinate system).
        """
        if img.mode != "RGB":
            img = img.convert("RGB")
        w, h = img.size

        canvas, r, left, top = self._letterbox(img)
        x = np.asarray(canvas, dtype=np.float32).transpose(2, 0, 1)[None] / 255.0

        out = self.session.run(None, {self.input_name: x})[0]
        if out.ndim != 3 or out.shape[-1] != 6:
            raise RuntimeError(
                f"Unexpected output shape {out.shape}; expected (1, N, 6). "
                "Make sure the model was exported from the doclayout-yolo "
                "(YOLOv10) package, not plain ultralytics."
            )

        dets = []
        for x1, y1, x2, y2, score, cls in out[0]:
            if score < conf:
                continue
            name = CLASSES.get(int(cls), str(int(cls)))
            if classes is not None and name not in classes:
                continue
            # Undo letterbox, clamp to page bounds
            bx1 = float(np.clip((x1 - left) / r, 0, w))
            by1 = float(np.clip((y1 - top) / r, 0, h))
            bx2 = float(np.clip((x2 - left) / r, 0, w))
            by2 = float(np.clip((y2 - top) / r, 0, h))
            dets.append(
                {
                    "class": name,
                    "confidence": round(float(score), 4),
                    "bbox_px": [round(bx1, 1), round(by1, 1), round(bx2, 1), round(by2, 1)],
                }
            )
        return dets
