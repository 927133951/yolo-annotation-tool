@echo off
chcp 65001 >nul
title YOLO标注与训练自动化工具

echo ============================================
echo    YOLO标注与训练自动化工具 - 一键部署
echo ============================================
echo.

REM 检查Python是否安装
python --version >nul 2>&1
if errorlevel 1 (
    echo [错误] 未检测到Python，请先安装Python 3.9+
    echo 下载地址: https://www.python.org/downloads/
    pause
    exit /b 1
)

echo [√] Python环境检测通过
echo.

REM 检查虚拟环境是否存在
if not exist ".venv" (
    echo [*] 正在创建虚拟环境...
    python -m venv .venv
    if errorlevel 1 (
        echo [错误] 虚拟环境创建失败
        pause
        exit /b 1
    )
    echo [√] 虚拟环境创建成功
    echo.
)

REM 激活虚拟环境
call .venv\Scripts\activate.bat

REM 检查是否需要安装依赖
pip show ultralytics >nul 2>&1
if errorlevel 1 (
    echo [*] 正在安装依赖包...
    echo 这可能需要几分钟，请耐心等待...
    echo.
    
    REM 升级pip
    python -m pip install --upgrade pip -q
    
    REM 安装PyTorch GPU版本
    echo [*] 安装PyTorch GPU版本...
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124 -q
    
    REM 安装ONNX Runtime GPU版本
    echo [*] 安装ONNX Runtime GPU版本...
    pip install onnxruntime-gpu==1.19.2 -q
    
    REM 安装其他依赖
    echo [*] 安装其他依赖...
    pip install -r requirements.txt -q
    
    echo [√] 依赖安装完成
    echo.
)

REM 检查CUDA是否可用
echo [*] 检查GPU环境...
python -c "import torch; print(f'[√] CUDA可用: {torch.cuda.is_available()}'); print(f'[√] GPU设备: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"无\"}')" 2>nul
if errorlevel 1 (
    echo [!] GPU检测失败，将使用CPU模式
)
echo.

echo ============================================
echo    正在启动程序...
echo ============================================
echo.

REM 启动主程序
python app.py

REM 程序退出后暂停
echo.
echo 程序已退出
pause
