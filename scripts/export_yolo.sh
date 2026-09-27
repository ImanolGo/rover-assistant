#!/bin/bash
# bench 1.2 prerequisite: export YOLO11n to TensorRT FP16 engine on the Jetson.
# Engines are not portable across TensorRT versions; always export on-device.
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate
python3 - <<'EOF'
from ultralytics import YOLO
model = YOLO("models/yolo11n.pt")
path = model.export(format="engine", half=True, imgsz=640, device=0, workspace=4)
print("exported:", path)
EOF
mkdir -p models/yolo_trt
mv models/yolo11n.engine models/yolo_trt/yolo11n_fp16.engine
ls -la models/yolo_trt/
