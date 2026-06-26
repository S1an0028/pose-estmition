import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO


HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
    (5, 9), (9, 13), (13, 17),
]

# Expected 21-point hand layout, matching the common MediaPipe/Ultralytics order:
# 0 wrist, then thumb/index/middle/ring/pinky with 4 joints per finger.
KEYPOINT_NAMES = [
    "wrist",
    "thumb_cmc", "thumb_mcp", "thumb_ip", "thumb_tip",
    "index_mcp", "index_pip", "index_dip", "index_tip",
    "middle_mcp", "middle_pip", "middle_dip", "middle_tip",
    "ring_mcp", "ring_pip", "ring_dip", "ring_tip",
    "pinky_mcp", "pinky_pip", "pinky_dip", "pinky_tip",
]

FINGER_COLORS = {
    "thumb": (80, 220, 255),
    "index": (80, 255, 120),
    "middle": (255, 210, 80),
    "ring": (255, 120, 220),
    "pinky": (180, 130, 255),
    "palm": (230, 230, 230),
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run YOLO hand keypoint detection on a local webcam."
    )
    parser.add_argument(
        "--model",
        default="models/yolo11n-pose-hands-community.pt",
        help="Path to a YOLO pose model trained for 21 hand keypoints.",
    )
    parser.add_argument("--camera", type=int, default=0, help="Webcam index.")
    parser.add_argument("--device", default="0", help="CUDA device, cpu, or 0.")
    parser.add_argument(
        "--profile",
        choices=("accurate", "balanced", "fast"),
        default="accurate",
        help="Preset for webcam quality and speed.",
    )
    parser.add_argument("--imgsz", type=int, default=None, help="Inference image size.")
    parser.add_argument("--conf", type=float, default=None, help="Box confidence threshold.")
    parser.add_argument(
        "--kpt-conf",
        type=float,
        default=None,
        help="Minimum confidence for drawing a keypoint.",
    )
    parser.add_argument(
        "--smooth",
        type=float,
        default=None,
        help="Temporal smoothing amount from 0.0 to 0.95. Higher is steadier but slower to react.",
    )
    parser.add_argument(
        "--max-hands",
        type=int,
        default=2,
        help="Maximum number of hands to draw.",
    )
    parser.add_argument(
        "--camera-width",
        type=int,
        default=1280,
        help="Requested webcam capture width.",
    )
    parser.add_argument(
        "--camera-height",
        type=int,
        default=720,
        help="Requested webcam capture height.",
    )
    parser.add_argument(
        "--augment",
        action="store_true",
        help="Use test-time augmentation. This can improve difficult frames but is slower.",
    )
    parser.add_argument(
        "--no-mirror",
        action="store_true",
        help="Do not horizontally mirror the webcam preview.",
    )
    parser.add_argument(
        "--label-all-points",
        action="store_true",
        help="Draw numbers for all 21 keypoints. Useful for checking keypoint order.",
    )
    parser.add_argument(
        "--save",
        default="",
        help="Optional output video path, for example runs/hand_webcam.avi.",
    )
    return parser.parse_args()


def apply_profile(args):
    # Presets keep common tuning choices in one place. You can still override
    # any value from the command line, for example `--imgsz 1280`.
    profiles = {
        "accurate": {"imgsz": 960, "conf": 0.25, "kpt_conf": 0.22, "smooth": 0.65},
        "balanced": {"imgsz": 768, "conf": 0.30, "kpt_conf": 0.25, "smooth": 0.55},
        "fast": {"imgsz": 640, "conf": 0.35, "kpt_conf": 0.28, "smooth": 0.35},
    }
    profile = profiles[args.profile]
    if args.imgsz is None:
        args.imgsz = profile["imgsz"]
    if args.conf is None:
        args.conf = profile["conf"]
    if args.kpt_conf is None:
        args.kpt_conf = profile["kpt_conf"]
    if args.smooth is None:
        args.smooth = profile["smooth"]
    args.smooth = min(max(args.smooth, 0.0), 0.95)
    return args


class TemporalSmoother:
    """Simple per-hand temporal smoothing to reduce webcam keypoint jitter."""

    def __init__(self, smooth_amount):
        self.smooth_amount = smooth_amount
        self.previous = []

    @staticmethod
    def center(box):
        return np.array([(box[0] + box[2]) * 0.5, (box[1] + box[3]) * 0.5], dtype=np.float32)

    @staticmethod
    def diagonal(box):
        return float(np.hypot(max(box[2] - box[0], 1.0), max(box[3] - box[1], 1.0)))

    def update(self, boxes, keypoints):
        # Match each current hand to the closest hand from the previous frame.
        # This is lighter than a full tracker and works well for webcam demos.
        if self.smooth_amount <= 0 or len(boxes) == 0:
            self.previous = [
                {"box": np.array(box, dtype=np.float32), "kpts": np.array(kpts, dtype=np.float32)}
                for box, kpts in zip(boxes, keypoints)
            ]
            return boxes, keypoints

        smoothed_boxes = []
        smoothed_keypoints = []
        used_previous = set()

        for box, kpts in zip(boxes, keypoints):
            box = np.array(box, dtype=np.float32)
            kpts = np.array(kpts, dtype=np.float32)
            current_center = self.center(box)

            best_idx = None
            best_distance = float("inf")
            for idx, previous in enumerate(self.previous):
                if idx in used_previous:
                    continue
                distance = float(np.linalg.norm(current_center - self.center(previous["box"])))
                allowed = max(self.diagonal(box), self.diagonal(previous["box"])) * 0.75
                if distance < best_distance and distance < allowed:
                    best_idx = idx
                    best_distance = distance

            if best_idx is None:
                # New hand or a hand that moved too far: do not blend with stale data.
                out_box = box
                out_kpts = kpts
            else:
                used_previous.add(best_idx)
                previous = self.previous[best_idx]
                keep = self.smooth_amount
                take = 1.0 - keep
                out_box = previous["box"] * keep + box * take
                out_kpts = previous["kpts"].copy()

                for point_idx in range(min(len(kpts), len(out_kpts))):
                    # Keep the newest confidence score, but smooth x/y position.
                    score = kpts[point_idx, 2] if kpts.shape[1] > 2 else 1.0
                    if score > 0:
                        out_kpts[point_idx, :2] = (
                            previous["kpts"][point_idx, :2] * keep + kpts[point_idx, :2] * take
                        )
                        if kpts.shape[1] > 2:
                            out_kpts[point_idx, 2] = kpts[point_idx, 2]

            smoothed_boxes.append(out_box)
            smoothed_keypoints.append(out_kpts)

        self.previous = [
            {"box": box.copy(), "kpts": kpts.copy()}
            for box, kpts in zip(smoothed_boxes, smoothed_keypoints)
        ]
        return np.array(smoothed_boxes), np.array(smoothed_keypoints)


def color_for_connection(a, b):
    if (a, b) in [(5, 9), (9, 13), (13, 17)]:
        return FINGER_COLORS["palm"]
    if a in range(1, 5) or b in range(1, 5):
        return FINGER_COLORS["thumb"]
    if a in range(5, 9) or b in range(5, 9):
        return FINGER_COLORS["index"]
    if a in range(9, 13) or b in range(9, 13):
        return FINGER_COLORS["middle"]
    if a in range(13, 17) or b in range(13, 17):
        return FINGER_COLORS["ring"]
    if a in range(17, 21) or b in range(17, 21):
        return FINGER_COLORS["pinky"]
    return FINGER_COLORS["palm"]


def get_visible_keypoints(kpts, threshold):
    visible = []
    for item in kpts:
        x = float(item[0])
        y = float(item[1])
        score = float(item[2]) if len(item) > 2 else 1.0
        visible.append((x, y, score, score >= threshold and x > 0 and y > 0))
    return visible


def draw_hand(frame, box, kpts, hand_id, box_conf, kpt_conf, label_all_points):
    x1, y1, x2, y2 = [int(v) for v in box]
    cv2.rectangle(frame, (x1, y1), (x2, y2), (70, 220, 255), 2)
    cv2.putText(
        frame,
        f"hand {hand_id} {box_conf:.2f}",
        (x1, max(24, y1 - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (70, 220, 255),
        2,
        cv2.LINE_AA,
    )

    # Convert model output to a compact list so drawing code stays simple.
    points = get_visible_keypoints(kpts, kpt_conf)

    # Draw skeleton lines before dots so the joints remain visible on top.
    for a, b in HAND_CONNECTIONS:
        ax, ay, _, a_ok = points[a]
        bx, by, _, b_ok = points[b]
        if a_ok and b_ok:
            cv2.line(
                frame,
                (int(ax), int(ay)),
                (int(bx), int(by)),
                color_for_connection(a, b),
                3,
                cv2.LINE_AA,
            )

    for idx, (x, y, score, ok) in enumerate(points):
        if not ok:
            continue
        center = (int(x), int(y))
        cv2.circle(frame, center, 5, (15, 15, 15), -1, cv2.LINE_AA)
        cv2.circle(frame, center, 3, (255, 255, 255), -1, cv2.LINE_AA)
        if label_all_points or idx in (0, 4, 8, 12, 16, 20):
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


def draw_status(frame, fps, hand_count, device_name, profile):
    text = f"{profile} | FPS {fps:4.1f} | hands {hand_count} | {device_name} | q/Esc quit"
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 34), (20, 20, 20), -1)
    cv2.putText(
        frame,
        text,
        (12, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (245, 245, 245),
        2,
        cv2.LINE_AA,
    )


def main():
    args = apply_profile(parse_args())
    model_path = Path(args.model)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    if args.device != "cpu" and not torch.cuda.is_available():
        print("CUDA is not available. Falling back to CPU.")
        args.device = "cpu"

    device_name = "CPU"
    if args.device != "cpu" and torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(int(args.device))

    model = YOLO(str(model_path))
    smoother = TemporalSmoother(args.smooth)

    cap = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open camera index {args.camera}. Try --camera 1 or check camera permissions."
        )

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.camera_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.camera_height)

    writer = None
    save_path = Path(args.save) if args.save else None
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)

    prev_time = time.perf_counter()
    fps = 0.0

    print("Webcam started. Press q or Esc in the preview window to quit.")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Camera frame read failed.")
                break

            if not args.no_mirror:
                frame = cv2.flip(frame, 1)

            results = model.predict(
                frame,
                imgsz=args.imgsz,
                conf=args.conf,
                device=args.device,
                augment=args.augment,
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

            hand_count = min(len(boxes), len(keypoints))
            if hand_count:
                order = np.argsort(box_conf[:hand_count])[::-1][: args.max_hands]
                boxes = boxes[order]
                keypoints = keypoints[order]
                box_conf = box_conf[order]
                boxes, keypoints = smoother.update(boxes, keypoints)
                hand_count = len(boxes)
            else:
                smoother.previous = []

            for idx in range(hand_count):
                draw_hand(
                    frame,
                    boxes[idx],
                    keypoints[idx],
                    idx + 1,
                    float(box_conf[idx]) if len(box_conf) > idx else 0.0,
                    args.kpt_conf,
                    args.label_all_points,
                )

            now = time.perf_counter()
            instant_fps = 1.0 / max(now - prev_time, 1e-6)
            fps = instant_fps if fps == 0.0 else fps * 0.9 + instant_fps * 0.1
            prev_time = now

            draw_status(frame, fps, hand_count, device_name, args.profile)

            if writer is None and save_path:
                fourcc = cv2.VideoWriter_fourcc(*"XVID")
                height, width = frame.shape[:2]
                writer = cv2.VideoWriter(str(save_path), fourcc, 30.0, (width, height))
            if writer is not None:
                writer.write(frame)

            cv2.imshow("YOLO Hand Keypoints", frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
