@echo off
setlocal enabledelayedexpansion
chcp 65001 >nul 2>&1
title YOLO Training Tool

cls
echo ============================================
echo    YOLO Training Tool - Auto Deploy
echo ============================================
echo.

REM ============================================
REM Step 1: Change to script directory
REM ============================================
cd /d "%~dp0"
echo Working directory: %cd%
echo.

REM ============================================
REM Step 2: Find Python
REM ============================================
echo [Step 1/6] Checking Python...
set "PYTHON_EXE="

REM Try common Python locations
for %%p in (
    "python"
    "python3"
    "%LOCALAPPDATA%\Programs\Python\Python312\python.exe"
    "%LOCALAPPDATA%\Programs\Python\Python311\python.exe"
    "%LOCALAPPDATA%\Programs\Python\Python310\python.exe"
    "%LOCALAPPDATA%\Programs\Python\Python39\python.exe"
) do (
    if not defined PYTHON_EXE (
        %%p --version >nul 2>&1
        if not errorlevel 1 (
            set "PYTHON_EXE=%%p"
        )
    )
)

if not defined PYTHON_EXE (
    echo [ERROR] Python not found!
    echo Please install Python 3.9+ from https://www.python.org/downloads/
    echo Make sure to check "Add Python to PATH" during installation.
    echo.
    goto :error_end
)

%PYTHON_EXE% --version
echo [OK] Python found
echo.

REM ============================================
REM Step 3: Create virtual environment
REM ============================================
echo [Step 2/6] Checking virtual environment...
if not exist ".venv\Scripts\python.exe" (
    echo Creating virtual environment...
    %PYTHON_EXE% -m venv .venv
    if errorlevel 1 (
        echo [ERROR] Failed to create virtual environment
        goto :error_end
    )
    echo [OK] Virtual environment created
) else (
    echo [OK] Virtual environment exists
)
echo.

REM ============================================
REM Step 4: Install dependencies
REM ============================================
echo [Step 3/6] Checking dependencies...
.venv\Scripts\python.exe -c "import PyQt5, torch, ultralytics" >nul 2>&1
if errorlevel 1 (
    echo Installing dependencies...
    echo This may take 5-10 minutes, please wait...
    echo.
    
    echo [1/4] Upgrading pip...
    .venv\Scripts\python.exe -m pip install --upgrade pip --quiet
    
    echo [2/4] Installing PyTorch GPU...
    .venv\Scripts\pip.exe install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124 --quiet
    
    echo [3/4] Installing ONNX Runtime GPU...
    .venv\Scripts\pip.exe install onnxruntime-gpu==1.19.2 --quiet
    
    echo [4/4] Installing other dependencies...
    .venv\Scripts\pip.exe install PyQt5 Pillow ultralytics onnx opencv-python numpy pandas matplotlib seaborn PyYAML tqdm --quiet
    
    echo [OK] Dependencies installed
) else (
    echo [OK] All dependencies installed
)
echo.

REM ============================================
REM Step 5: Check GPU
REM ============================================
echo [Step 4/6] Checking GPU...
.venv\Scripts\python.exe -c "import torch; print('CUDA:', torch.cuda.is_available(), '| GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"
echo.

REM ============================================
REM Step 6: Check app.py
REM ============================================
echo [Step 5/6] Checking application files...
if not exist "app.py" (
    echo [ERROR] app.py not found!
    echo Make sure run.bat is in the same folder as app.py
    goto :error_end
)
echo [OK] Application files found
echo.

REM ============================================
REM Step 7: Run application
REM ============================================
echo [Step 6/6] Starting application...
echo ============================================
echo.

.venv\Scripts\python.exe app.py
set APP_EXIT_CODE=%errorlevel%

echo.
if %APP_EXIT_CODE% neq 0 (
    echo [WARNING] Application exited with code: %APP_EXIT_CODE%
)
echo ============================================
echo Application closed.
echo ============================================
echo.
goto :end

:error_end
echo.
echo ============================================
echo An error occurred. Please check the messages above.
echo ============================================
echo.

:end
echo Press any key to close this window...
pause >nul
exit /b 0
