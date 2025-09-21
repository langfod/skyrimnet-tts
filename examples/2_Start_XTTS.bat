@echo off
cls

ECHO   ad88888ba   88                                 88                      888b      88                       
ECHO  d8"     "8b  88                                 ""                      8888b     88                ,d     
ECHO  Y8,          88                                                         88 `8b    88                88     
ECHO  `Y8aaaaa,    88   ,d8  8b       d8  8b,dPPYba,  88  88,dPYba,,adPYba,   88  `8b   88   ,adPPYba,  MM88MMM  
ECHO   `""""""8b,  88 ,a8"   `8b     d8'  88P'   "Y8  88  88P'   "88"    "8a  88   `8b  88  a8P_____88    88     
ECHO          `8b  8888[      `8b   d8'   88          88  88      88      88  88    `8b 88  8PP"""""""    88     
ECHO  Y8a     a8P  88`"Yba,    `8b,d8'    88          88  88      88      88  88     `8888  "8b,   ,aa    88,    
ECHO   "Y88888P"   88   `Y8a     Y88'     88          88  88      88      88  88      `888   `"Ybbd8"'    "Y888  
ECHO                             d8'                                               
ECHO                            d8'       XTTS (XTTS / Gradio / Zonos Emulated)                                       
echo.

echo Attempting to start SkyrimNet XTTS...

REM Get the directory where this batch file is located
set "SCRIPT_DIR=%~dp0"

REM Check for the executable
set "EXE_PATH=%SCRIPT_DIR%skyrimnet-xtts.exe"
if not exist "%EXE_PATH%" (
    echo ERROR: Could not find executable: %EXE_PATH%
    echo.
    pause
    exit /b 1
)

echo Using executable: %EXE_PATH%

REM Check for CPU mode argument
set "EXE_ARGS="
if /i "%1"=="-cpu" (
    set "EXE_ARGS=--cpu"
    echo CPU mode enabled
)
if /i "%1"=="--cpu" (
    set "EXE_ARGS=--cpu"
    echo CPU mode enabled
)

REM Display what we're about to run
if defined EXE_ARGS (
    echo Starting new window to run: %EXE_PATH% %EXE_ARGS%
) else (
    echo Starting new window to run: %EXE_PATH%
)

REM Start the executable in a new window with high priority
start /high "SkyrimNet XTTS" "%EXE_PATH%" %EXE_ARGS%

echo.
echo SkyrimNet XTTS should start in another window. Default web server is http://localhost:7860
echo If that window closes immediately, run 'skyrimnet-xtts.exe' directly to capture errors.
echo.
echo Otherwise, you may close this window if it does not close itself.

REM Wait a bit then exit
timeout /t 20 /nobreak >nul 2>&1