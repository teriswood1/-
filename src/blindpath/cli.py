from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from .config import load_models
from .inference import GuidanceEngine


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="BlindPath real-time YOLO guidance prototype")
    parser.add_argument("--config", default="configs/models.yaml")
    parser.add_argument("--source", default="0", help="camera index or video path")
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--speak", action="store_true", help="enable optional offline text-to-speech")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = Path(args.config)
    if not config_path.exists():
        raise SystemExit(f"找不到 {config_path}。请先复制 configs/models.example.yaml 并填写权重路径。")
    source: int | str = int(args.source) if args.source.isdigit() else args.source
    capture = cv2.VideoCapture(source)
    if not capture.isOpened():
        raise SystemExit(f"无法打开视频源：{args.source}")
    engine = GuidanceEngine(load_models(config_path))
    speaker = None
    if args.speak:
        try:
            import pyttsx3
            speaker = pyttsx3.init()
        except ImportError as exc:
            raise SystemExit("语音播报需要安装 pyttsx3：pip install pyttsx3") from exc
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            detections = engine.detect(frame)
            if message := engine.guidance(detections):
                print(f"[GUIDANCE] {message}")
                if speaker:
                    speaker.say(message)
                    speaker.runAndWait()
            if args.show:
                cv2.imshow("BlindPath", frame)
                if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                    break
    finally:
        capture.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
