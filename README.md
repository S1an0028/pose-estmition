# Hand Pose Estimation Workspace

This workspace is configured for YOLO pose experiments with Ultralytics.

## Install From GitHub

```powershell
git clone https://github.com/S1an0028/pose-estmition.git
cd pose-estmition
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

The repository includes the two small model weight files used by the app:

- `models/yolo11n-pose-hands-community.pt`
- `yolo11n-pose.pt`

An NVIDIA GPU is recommended. If CUDA is unavailable, the browser app falls
back to CPU, but inference will be slower.

## Activate Environment

```powershell
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation scripts, use the Python executable directly:

```powershell
.\.venv\Scripts\python.exe check_env.py
```

## Verify GPU and YOLO

```powershell
.\.venv\Scripts\python.exe check_env.py
```

## Run Webcam Hand Keypoints

```powershell
.\.venv\Scripts\python.exe hand_webcam_yolo.py
```

The default profile is `accurate`, which uses a larger inference size and
temporal smoothing for steadier hand keypoints.

Controls:

- Press `q` or `Esc` in the preview window to quit.
- If your webcam is not camera `0`, try `--camera 1` or `--camera 2`.

Examples:

```powershell
.\.venv\Scripts\python.exe hand_webcam_yolo.py --camera 1
.\.venv\Scripts\python.exe hand_webcam_yolo.py --profile balanced
.\.venv\Scripts\python.exe hand_webcam_yolo.py --profile fast
.\.venv\Scripts\python.exe hand_webcam_yolo.py --imgsz 1280 --conf 0.25 --kpt-conf 0.2 --smooth 0.75
.\.venv\Scripts\python.exe hand_webcam_yolo.py --conf 0.45 --kpt-conf 0.4
.\.venv\Scripts\python.exe hand_webcam_yolo.py --label-all-points
.\.venv\Scripts\python.exe hand_webcam_yolo.py --no-mirror
.\.venv\Scripts\python.exe hand_webcam_yolo.py --augment
.\.venv\Scripts\python.exe hand_webcam_yolo.py --save runs/hand_webcam.avi
.\.venv\Scripts\python.exe hand_webcam_yolo.py --model path/to/your/best.pt
```

Accuracy tips:

- Use `--profile accurate` or the default command first.
- If the points are jittery, increase `--smooth` toward `0.75`.
- If wrong points are drawn, raise thresholds, for example `--conf 0.45 --kpt-conf 0.4`.
- If small finger joints are missed, try `--imgsz 1280`, but expect lower FPS.
- If finger connections look wrong, run with `--label-all-points` and check where keypoint numbers 0-20 land.
- If right/left hand orientation still looks wrong, compare default mode with `--no-mirror`.
- If the program becomes too slow, use `--profile balanced` or `--profile fast`.
- Use a plain background, keep the hand large in frame, and avoid heavy motion blur.

## Run Browser Version

The original desktop command above still works. The browser version is an
additional local web app that uses the same environment. It can switch between
the hand keypoint model and the native YOLO11 body pose model.

```powershell
.\.venv\Scripts\python.exe web_hand_yolo.py
```

You can also double-click:

```text
start_web_yolo.bat
```

Then open:

```text
http://127.0.0.1:7860
```

Or double-click `index.html` in this folder and use the `打开网页应用` button.

Click `开始` and allow camera access in the browser. Keep the PowerShell window
running while using the page.

In the page:

- Use `检测类型` to switch between `手部 21 点` and `全身人体姿态`.
- Use `摄像头` to select the built-in webcam or an external USB camera.
- If the external camera does not appear, click `刷新摄像头` after allowing camera access.

To stop a background web server on port `7860`, double-click:

```text
stop_web_yolo.bat
```

## Current Core Packages

- Python 3.13.7
- PyTorch 2.11.0+cu128
- Torchvision 0.26.0+cu128
- Ultralytics 8.4.75

## Notes

Ultralytics is the Python package that provides the YOLO interface. It handles
model loading, prediction, training, validation, export, and command-line tools.
For YOLO pose models, PyTorch performs the underlying neural network computation.

The downloaded `yolo11n-pose.pt` model is an official YOLO11 pose model. It is
useful for testing the pose pipeline, but standard pretrained YOLO pose models
are primarily for human body keypoints. For hand and finger joints, we will need
a hand-keypoint model or a custom-trained YOLO pose model.

The webcam demo currently uses `models/yolo11n-pose-hands-community.pt`, a
community-trained YOLO11n pose model for 21 hand keypoints. It is useful for a
first prototype, but final quality may require training our own model.
