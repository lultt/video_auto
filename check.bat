@echo off
chcp 65001 >nul

echo ======================================== > env_report.txt
echo      Conda Environment Check Report     >> env_report.txt
echo ======================================== >> env_report.txt
echo. >> env_report.txt

echo Checking all conda environments...
echo.

for /f "tokens=1" %%i in ('conda env list ^| findstr /V "#"') do (

    echo ========================================
    echo ENV: %%i
    echo ========================================

    echo ======================================== >> env_report.txt
    echo ENV: %%i >> env_report.txt
    echo ======================================== >> env_report.txt

    call conda activate %%i

    echo Python: >> env_report.txt
    python --version >> env_report.txt 2>&1

    echo. >> env_report.txt
    echo OpenCV: >> env_report.txt
    python -c "import cv2; print(cv2.__version__)" >> env_report.txt 2>&1

    echo. >> env_report.txt
    echo OpenCV CUDA: >> env_report.txt
    python -c "import cv2; print(cv2.cuda.getCudaEnabledDeviceCount())" >> env_report.txt 2>&1

    echo. >> env_report.txt
    echo Torch CUDA: >> env_report.txt
    python -c "import torch; print(torch.cuda.is_available())" >> env_report.txt 2>&1

    echo. >> env_report.txt
    echo Torch GPU: >> env_report.txt
    python -c "import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No GPU')" >> env_report.txt 2>&1

    echo. >> env_report.txt
    echo Decord: >> env_report.txt
    python -c "import decord; print(decord.__version__)" >> env_report.txt 2>&1

    echo. >> env_report.txt
    echo PyArrow: >> env_report.txt
    python -c "import pyarrow; print(pyarrow.__version__)" >> env_report.txt 2>&1

    echo. >> env_report.txt
    echo FFmpeg: >> env_report.txt
    ffmpeg -version >> env_report.txt 2>&1

    echo. >> env_report.txt
    echo HW Accels: >> env_report.txt
    ffmpeg -hwaccels >> env_report.txt 2>&1

    echo. >> env_report.txt
    echo ---------------------------------------- >> env_report.txt
    echo. >> env_report.txt
)

echo.
echo ========================================
echo Finished
echo Report saved:
echo env_report.txt
echo ========================================

pause