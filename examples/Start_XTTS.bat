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



REM Check for help arguments first
if /i "%1"=="-help" goto show_help
if /i "%1"=="--help" goto show_help
if /i "%1"=="-h" goto show_help

echo Attempting to start SkyrimNet XTTS...

REM Set default values
set HOST="0.0.0.0"
set PORT=7860
set CPU_MODE=0

REM Parse command line arguments
:parse_args
if "%1"=="" goto parse_done

if /i "%1"=="-server" (
    set SERVER_MODE=1
)

if /i "%1"=="--server" (
    set SERVER_MODE=1
)

if defined SERVER_MODE (
    if "%2"=="" (
        echo ERROR: Server parameter requires a host address
        goto show_help
    )
    set "HOST=%2"
    shift
    shift
    set SERVER_MODE=
    goto parse_args
)

if /i "%1"=="-port" (
    set PORT_MODE=1
)

if /i "%1"=="--port" (
    set PORT_MODE=1
)

if defined PORT_MODE (
    if "%2"=="" (
        echo ERROR: Port parameter requires a port number
        goto show_help
    )
    REM Validate port number
    echo %2| findstr /r "^[0-9][0-9]*$" >nul
    if errorlevel 1 (
        echo ERROR: Port must be a valid number, got: %2
        goto show_help
    )
    REM Check port range (1-65535)
    if %2 LSS 1 (
        echo ERROR: Port must be between 1 and 65535, got: %2
        goto show_help
    )
    if %2 GTR 65535 (
        echo ERROR: Port must be between 1 and 65535, got: %2
        goto show_help
    )
    set PORT=%2
    shift
    shift
    set PORT_MODE=
    goto parse_args
)

if /i "%1"=="-cpu" (
    set CPU_MODE=1
    shift
    goto parse_args
)

if /i "%1"=="--cpu" (
    set CPU_MODE=1
    shift
    goto parse_args
)

REM Unknown argument
echo ERROR: Unknown argument: %1
goto show_help

:parse_done

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
echo Server: %HOST%
echo Port: %PORT%

REM Build arguments
set "EXE_ARGS=--server %HOST% --port %PORT%"

if %CPU_MODE%==1 (
    set "EXE_ARGS=%EXE_ARGS% --cpu"
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
echo SkyrimNet XTTS should start in another window. Default web server is http://localhost:%PORT%
echo If that window closes immediately, run 'skyrimnet-xtts.exe' directly to capture errors.
echo.
echo Otherwise, you may close this window if it does not close itself.

REM Wait a bit then exit
timeout /t 5 /nobreak >nul 2>&1
timeout /t 15 >nul 2>&1

goto :eof

REM Function to show help
:show_help
echo.
echo USAGE: Start_XTTS.bat [OPTIONS]
echo.
echo OPTIONS:
echo   -server (or --server) HOST     Server host address (default: "0.0.0.0")
echo   -port (or --port) NUMBER     Server port number (default: 7860)
echo   -cpu (or --cpu)            Enable CPU mode
echo   -help (or --help)         Show this help message
echo   -h (or --h)               Show this help message
echo.
echo EXAMPLES:
echo   Start_XTTS.bat
echo   Start_XTTS.bat -server "127.0.0.1" -port 8080
echo   Start_XTTS.bat --server "127.0.0.1" --port 8080
echo   Start_XTTS.bat -cpu -port 9090
echo   Start_XTTS.bat --cpu --port 9090
echo.
pause

:eof
exit /b 0