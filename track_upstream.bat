@echo off
REM Track upstream changes in COQUI_AI_TTS
REM This script compares our local copy against upstream GitHub repository

echo.
echo COQUI_AI_TTS Upstream Tracker
echo =============================
echo.

REM Install requirements if needed
python -m pip install -r track_upstream_requirements.txt -q

REM Run the tracking script
python track_upstream_changes.py --baseline v0.27.1 --target v0.27.2 --output upstream_report.md

echo.
echo Report generated: upstream_report.md
echo.
pause