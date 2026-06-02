@echo off
chcp 65001 >nul
cd /d "%~dp0"

echo ========================================
echo   从 GitHub 下载最新代码 (lultt/video_auto)
echo ========================================
echo.

echo 正在拉取最新代码...
git pull
if errorlevel 1 (
    echo.
    echo [错误] 下载失败。可能是本地有未保存的改动和远程冲突，
    echo        或网络/SSH 问题。请先用 save.bat 保存本地改动再试。
) else (
    echo.
    echo [成功] 已是最新代码！
)

echo.
pause
