param(
    [switch]$test,
    [switch]$nobuild,
    [switch]$noarchive,
    [switch]$noclean
)

$PACKAGE_NAME = "SkyrimNet_XTTS"

if (-not $nobuild -or $noclean) {
    
    if (-not (Test-Path ".venv\Scripts\Activate.ps1")) {
        Write-Host "Virtual environment not found. Please set up the virtual environment before building." -ForegroundColor Red
        exit 1
    }
    & .venv\Scripts\Activate.ps1
    if (-not (Get-Command pyinstaller -ErrorAction SilentlyContinue)) {
        & pip install pyinstaller
    }
    Write-Host "Starting build process..."
    if ($noclean) {
        Write-Host "Skipping clean step as per -noclean flag."
        & pyinstaller --noconfirm --log-level=WARN skyrimnet-xtts.spec
    }
    else {
        if (Test-Path "build") {
            Remove-Item -Path "build" -Recurse -Force
        }
        if (Test-Path "dist") {
            Remove-Item -Path "dist" -Recurse -Force
        }
        if (Test-Path "__pycache__") {
            Remove-Item -Path "__pycache__" -Recurse -Force
        }
        & pyinstaller --clean --noconfirm --log-level=ERROR skyrimnet-xtts.spec
    }
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Build failed. Exiting."
        exit $LASTEXITCODE
    }
    Deactivate
}

if ($test) {
    Write-Host "Running in test mode: Archive will be created but not deployed."
    if (-not (Test-Path "dist\skyrimnet-xtts\skyrimnet-xtts.exe")) {
        Write-Host "Error: Executable not found. Please build the project first."
        exit 1
    }

    Copy-Item -Path "models" -Destination "dist\skyrimnet-xtts\" -Force -Recurse
    Copy-Item -Path "speakers" -Destination "dist\skyrimnet-xtts\" -Force -Recurse
    Copy-Item -Path "assets" -Destination "dist\skyrimnet-xtts\" -Force -Recurse
    Copy-Item -Path "skyrimnet_config.txt" -Destination "dist\skyrimnet-xtts\" -Force -Recurse
    Copy-Item -Path "examples\Start.bat" -Destination "dist\skyrimnet-xtts\" -Force -Recurse
    Copy-Item -Path "examples\Start_XTTS.ps1" -Destination "dist\skyrimnet-xtts\" -Force -Recurse


    Set-Location -Path ./dist/skyrimnet-xtts
    & ./Start.bat -server "localhost" -port 7860
    Set-Location -Path ../..
}
else {
    Write-Host "Running in deployment mode: Archive will be created and deployed."
    if (-not (Test-Path "dist\skyrimnet-xtts\skyrimnet-xtts.exe")) {
        Write-Host "Error: Executable not found. Please build the project first."
        exit 1
    }

    if (Test-Path "archive") {
        Remove-Item -Path "archive" -Recurse -Force
    }
    New-Item -ItemType Directory -Path "archive/$PACKAGE_NAME" -Force
    New-Item -ItemType Directory -Path "archive/$PACKAGE_NAME/assets" -Force

    Get-ChildItem -Path "speakers" -Directory | Copy-Item -Destination "archive/$PACKAGE_NAME/speakers" 

    Copy-Item -Path "speakers\en\malebrute.wav" -Destination "archive/$PACKAGE_NAME/speakers/en\" -Force -Recurse
    Copy-Item -Path "speakers\en\malecommoner.wav" -Destination "archive/$PACKAGE_NAME/speakers/en\" -Force -Recurse
    Copy-Item -Path "assets\silence_100ms.wav" -Destination "archive/$PACKAGE_NAME/assets\" -Force -Recurse
    Copy-Item -Path "skyrimnet_config.txt" -Destination "archive/$PACKAGE_NAME\" -Force 
    Copy-Item -Path "examples\Start.bat" -Destination "archive/$PACKAGE_NAME\" -Force
    Copy-Item -Path "examples\Start_XTTS.ps1" -Destination "archive/$PACKAGE_NAME\" -Force
    Copy-Item -Path "dist\skyrimnet-xtts\skyrimnet-xtts.exe" -Destination "archive/$PACKAGE_NAME\" -Force
    Copy-Item -Path "dist\skyrimnet-xtts\_internal" -Destination "archive/$PACKAGE_NAME\" -Force -Recurse


    if (-not $noarchive) {
        $archiveName = "$PACKAGE_NAME"
        # extract version from pyproject.toml if needed from first occurance of version = "0.1.0"
        $version = Select-String -Path "pyproject.toml" -Pattern 'version = "(.*)"' | ForEach-Object { $_.Matches.Groups[1].Value }
        if ($version) {
            $archiveName += "_$version"
        }
        Write-Host "Creating archive: $archiveName.zip"
        Set-Location -Path ./archive
        & "C:\Program Files\7-Zip\7z.exe" a -t7z "$archiveName.7z" "$PACKAGE_NAME" -mx=9
    }
}
