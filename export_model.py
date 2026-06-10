"""One-time setup: download DocStructBench weights and export to ONNX.

Needs the heavy deps (torch, doclayout-yolo) only for this step --
do it in a throwaway venv, keep just model.onnx, delete the venv.

  python -m venv /tmp/export-venv && source /tmp/export-venv/bin/activate
  pip install doclayout-yolo huggingface_hub onnx onnxslim
  python export_model.py
  deactivate && rm -rf /tmp/export-venv
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
