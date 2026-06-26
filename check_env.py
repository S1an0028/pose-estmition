from ultralytics import YOLO
import torch


def main():
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"CUDA runtime: {torch.version.cuda}")

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        x = torch.randn(1024, 1024, device="cuda")
        y = x @ x
        print(f"GPU compute OK: {tuple(y.shape)}")

    model = YOLO("yolo11n-pose.pt")
    print(f"Ultralytics YOLO task: {model.task}")
    print("YOLO11 pose model load OK")


if __name__ == "__main__":
    main()
