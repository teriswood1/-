"""Train a YOLO model from a local dataset configuration."""

from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="YOLO data.yaml path")
    parser.add_argument("--model", default="yolo11n.pt")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="0")
    parser.add_argument("--project", default="runs/train")
    parser.add_argument("--name", default="blindpath-exp")
    args = parser.parse_args()
    if not Path(args.data).exists():
        raise SystemExit(f"找不到数据配置：{args.data}")
    YOLO(args.model).train(data=args.data, epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
                           device=args.device, project=args.project, name=args.name)


if __name__ == "__main__":
    main()
