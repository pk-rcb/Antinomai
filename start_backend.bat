@echo off
echo Starting Antinomai Backend...
cd /d %~dp0
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload

