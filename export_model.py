"""One-time setup: download DocStructBench weights and export to ONNX.

Needs the heavy deps (torch, doclayout-yolo) only for this step -- run it in a
throwaway uv env and keep just model.onnx. onnxscript is required by torch's
ONNX exporter (without it: "No module named 'onnxscript'").

  uv run --no-project --python 3.12 \
    --with doclayout-yolo --with huggingface_hub --with onnx --with onnxslim --with onnxscript \
    python export_model.py
"""

import shutil

from doclayout_yolo import YOLOv10
from huggingface_hub import hf_hub_download

ckpt = hf_hub_download(
    "juliozhao/DocLayout-YOLO-DocStructBench",
    "doclayout_yolo_docstructbench_imgsz1024.pt",
)

model = YOLOv10(ckpt)
onnx_path = model.export(format="onnx", imgsz=1024)
shutil.move(onnx_path, "model.onnx")
print("wrote model.onnx")
