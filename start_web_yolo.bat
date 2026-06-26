@echo off
cd /d "%~dp0"
.\.venv\Scripts\python.exe web_hand_yolo.py --host 127.0.0.1 --port 7860
