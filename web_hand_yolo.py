import argparse
import base64
import time
from pathlib import Path
from threading import Lock

import cv2
import numpy as np
import torch
from flask import Flask, jsonify, request, send_from_directory
from ultralytics import YOLO
from waitress import serve

from hand_webcam_yolo import TemporalSmoother, apply_profile, draw_hand


ROOT = Path(__file__).resolve().parent
STATIC_DIR = ROOT / "web_static"
HAND_MODEL_PATH = ROOT / "models" / "yolo11n-pose-hands-community.pt"
BODY_MODEL_PATH = ROOT / "yolo11n-pose.pt"

BODY_CONNECTIONS = [
    (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 6), (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
    (0, 1), (0, 2), (1, 3), (2, 4),
]

BODY_KEYPOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]

app = Flask(__name__, static_folder=str(STATIC_DIR), static_url_path="/static")

# The web app keeps both models loaded so switching between hand and body mode
# does not reload weights on every request.
models = {}
model_lock = Lock()
smoothers = {
    "hand": TemporalSmoother(0.65),
    "body": TemporalSmoother(0.65),
}
last_smooth = {
    "hand": 0.65,
    "body": 0.65,
}


def parse_args():
    parser = argparse.ArgumentParser(description="Run the browser YOLO pose app.")
    parser.add_argument("--host", default="127.0.0.1", help="Web server host.")
    parser.add_argument("--port", type=int, default=7860, help="Web server port.")
    parser.add_argument("--device", default="0", help="CUDA device, cpu, or 0.")
    parser.add_argument("--hand-model", default=str(HAND_MODEL_PATH), help="YOLO hand keypoint model path.")
    parser.add_argument("--body-model", default=str(BODY_MODEL_PATH), help="YOLO11 body pose model path.")
    return parser.parse_args()


def load_model(model_path):
    if not Path(model_path).exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    return YOLO(model_path)


def decode_image(data_url):
    """Decode a browser canvas data URL into an OpenCV BGR image."""
    if "," in data_url:
        data_url = data_url.split(",", 1)[1]
    image_bytes = base64.b64decode(data_url)
    image_array = np.frombuffer(image_bytes, dtype=np.uint8)
    frame = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError("Could not decode image data.")
    return frame


def encode_image(frame):
    ok, encoded = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    if not ok:
        raise ValueError("Could not encode result image.")
    return "data:image/jpeg;base64," + base64.b64encode(encoded).decode("ascii")


def make_profile(profile, imgsz, conf, kpt_conf, smooth):
    """Reuse the desktop script's profile defaults, then apply web overrides."""
    args = argparse.Namespace(
        profile=profile,
        imgsz=imgsz,
        conf=conf,
        kpt_conf=kpt_conf,
        smooth=smooth,
    )
    return apply_profile(args)


def get_visible_keypoints(kpts, threshold):
    visible = []
    for item in kpts:
        x = float(item[0])
        y = float(item[1])
        score = float(item[2]) if len(item) > 2 else 1.0
        visible.append((x, y, score, score >= threshold and x > 0 and y > 0))
    return visible


def draw_body(frame, box, kpts, person_id, box_conf, kpt_conf, label_all_points):
    x1, y1, x2, y2 = [int(v) for v in box]
    cv2.rectangle(frame, (x1, y1), (x2, y2), (90, 180, 255), 2)
    cv2.putText(
        frame,
        f"person {person_id} {box_conf:.2f}",
        (x1, max(24, y1 - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (90, 180, 255),
        2,
        cv2.LINE_AA,
    )

    points = get_visible_keypoints(kpts, kpt_conf)

    for a, b in BODY_CONNECTIONS:
        ax, ay, _, a_ok = points[a]
        bx, by, _, b_ok = points[b]
        if a_ok and b_ok:
            cv2.line(frame, (int(ax), int(ay)), (int(bx), int(by)), (90, 240, 180), 3, cv2.LINE_AA)

    for idx, (x, y, _, ok) in enumerate(points):
        if not ok:
            continue
        center = (int(x), int(y))
        cv2.circle(frame, center, 5, (15, 15, 15), -1, cv2.LINE_AA)
        cv2.circle(frame, center, 3, (255, 255, 255), -1, cv2.LINE_AA)
        if label_all_points or idx in (0, 5, 6, 9, 10, 15, 16):
            cv2.putText(
                frame,
                str(idx),
                (center[0] + 6, center[1] - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )


@app.get("/")
def index():
    return send_from_directory(STATIC_DIR, "index.html")


@app.post("/api/reset")
def reset():
    mode = request.get_json(silent=True) or {}
    selected = mode.get("mode")
    if selected in smoothers:
        smoothers[selected] = TemporalSmoother(last_smooth[selected])
    else:
        for key in smoothers:
            smoothers[key] = TemporalSmoother(last_smooth[key])
    return jsonify({"ok": True})


@app.post("/api/predict")
def predict():
    started = time.perf_counter()
    payload = request.get_json(force=True)
    frame = decode_image(payload["image"])

    # The browser sends one frame at a time and chooses which loaded YOLO model
    # should process it. This keeps the HTML simple while Python uses the GPU.
    mode = payload.get("mode", "hand")
    if mode not in models:
        return jsonify({"ok": False, "error": f"Unknown mode: {mode}"}), 400

    settings = make_profile(
        payload.get("profile", "accurate"),
        payload.get("imgsz"),
        payload.get("conf"),
        payload.get("kptConf"),
        payload.get("smooth"),
    )
    max_items = int(payload.get("maxItems", payload.get("maxHands", 2)))
    label_all_points = bool(payload.get("labelAllPoints", False))
    augment = bool(payload.get("augment", False))
    device = payload.get("device", app.config["DEVICE"])

    if settings.smooth != last_smooth[mode]:
        smoothers[mode] = TemporalSmoother(settings.smooth)
        last_smooth[mode] = settings.smooth

    with model_lock:
        # Ultralytics/PyTorch inference is guarded by a lock because the web
        # client may send the next frame while the previous one is still running.
        results = models[mode].predict(
            frame,
            imgsz=settings.imgsz,
            conf=settings.conf,
            device=device,
            augment=augment,
            verbose=False,
        )

    result = results[0]
    boxes = result.boxes.xyxy.cpu().numpy() if result.boxes is not None else []
    box_conf = result.boxes.conf.cpu().numpy() if result.boxes is not None else []
    keypoints = (
        result.keypoints.data.cpu().numpy()
        if result.keypoints is not None and result.keypoints.data is not None
        else []
    )

    detection_count = min(len(boxes), len(keypoints))
    if detection_count:
        order = np.argsort(box_conf[:detection_count])[::-1][:max_items]
        boxes = boxes[order]
        keypoints = keypoints[order]
        box_conf = box_conf[order]
        boxes, keypoints = smoothers[mode].update(boxes, keypoints)
        detection_count = len(boxes)
    else:
        smoothers[mode].previous = []

    for idx in range(detection_count):
        drawer = draw_hand if mode == "hand" else draw_body
        drawer(
            frame,
            boxes[idx],
            keypoints[idx],
            idx + 1,
            float(box_conf[idx]) if len(box_conf) > idx else 0.0,
            settings.kpt_conf,
            label_all_points,
        )

    elapsed_ms = round((time.perf_counter() - started) * 1000, 1)
    return jsonify(
        {
            "ok": True,
            "image": encode_image(frame),
            "mode": mode,
            "detections": int(detection_count),
            "hands": int(detection_count) if mode == "hand" else 0,
            "people": int(detection_count) if mode == "body" else 0,
            "elapsedMs": elapsed_ms,
            "imgsz": int(settings.imgsz),
            "conf": float(settings.conf),
            "kptConf": float(settings.kpt_conf),
            "smooth": float(settings.smooth),
        }
    )


def main():
    args = parse_args()
    device = args.device
    print("Starting browser YOLO app...", flush=True)
    if device != "cpu" and not torch.cuda.is_available():
        print("CUDA is not available. Falling back to CPU.", flush=True)
        device = "cpu"

    app.config["DEVICE"] = device

    global models
    print("Loading hand model...", flush=True)
    hand_model = load_model(args.hand_model)
    print("Loading body model...", flush=True)
    body_model = load_model(args.body_model)
    models = {
        "hand": hand_model,
        "body": body_model,
    }

    print(f"Browser app ready: http://{args.host}:{args.port}", flush=True)
    print(f"Hand model: {args.hand_model}", flush=True)
    print(f"Body model: {args.body_model}", flush=True)
    print("Keep this window running while using the web page.", flush=True)
    serve(app, host=args.host, port=args.port, threads=1)


if __name__ == "__main__":
    main()
