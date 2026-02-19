# #!/usr/bin/env python3
# # SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# # SPDX-License-Identifier: Apache-2.0
# """
# This script exports a set of YOLOv8 ONNX models for ADI workloads. The models are exported with fixed input shapes and opset 18, suitable for Polaris.

# Each is exported as:
#   workloads/onnx/<workload_name>.onnx
# """

# from pathlib import Path

# from ultralytics import YOLO

# OPSET = 18

# ROOT = Path(__file__).resolve().parents[3] 
# ONNX_DIR = ROOT / "workloads" / "onnx" 
# ONNX_DIR.mkdir(parents=True, exist_ok=True)

# def export_yolov8_adi() -> None:
#     # 1) yolov8s_detection_640x640
#     y8s = YOLO("yolov8s.pt")
#     tmp = Path(y8s.export(format="onnx", imgsz=(640, 640), opset=OPSET))
#     (ONNX_DIR / "yolov8s_detection_640x640.onnx").write_bytes(tmp.read_bytes())
#     tmp.unlink()

#     # 2) yolov8l_detection_640x640
#     y8l_det = YOLO("yolov8l.pt")
#     tmp = Path(y8l_det.export(format="onnx", imgsz=(640, 640), opset=OPSET))
#     (ONNX_DIR / "yolov8l_detection_640x640.onnx").write_bytes(tmp.read_bytes())
#     tmp.unlink()

#     # 3) yolov8l_detection_4kx4k (4096x4096)
#     tmp = Path(y8l_det.export(format="onnx", imgsz=(4096, 4096), opset=OPSET))
#     (ONNX_DIR / "yolov8l_detection_4kx4k.onnx").write_bytes(tmp.read_bytes())
#     tmp.unlink()

#     # 4) yolov8l_segmentation_640x640
#     y8l_seg = YOLO("yolov8l-seg.pt")
#     tmp = Path(y8l_seg.export(format="onnx", imgsz=(640, 640), opset=OPSET))
#     (ONNX_DIR / "yolov8l_segmentation_640x640.onnx").write_bytes(tmp.read_bytes())
#     tmp.unlink()

#     # 5) yolov8l_segmentation_4kx4k
#     tmp = Path(y8l_seg.export(format="onnx", imgsz=(4096, 4096), opset=OPSET))
#     (ONNX_DIR / "yolov8l_segmentation_4kx4k.onnx").write_bytes(tmp.read_bytes())
#     tmp.unlink()

#     print(f"Exported YOLOv8 ADI ONNX models (opset={OPSET}) under {ONNX_DIR}")

# if __name__ == "__main__":
#     export_yolov8_adi()