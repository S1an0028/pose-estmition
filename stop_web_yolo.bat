@echo off
for /f "tokens=5" %%p in ('netstat -ano ^| findstr ":7860" ^| findstr "LISTENING"') do (
  taskkill /PID %%p /F
)
