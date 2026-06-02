@echo off
chcp 65001 >nul
cd /d "%~dp0"

echo ========================================
echo   保存代码到 GitHub (lultt/video_auto)
echo ========================================
echo.

echo 当前改动：
git status --short
echo.

set "msg="
set /p "msg=请输入本次修改说明 (直接回车用默认时间戳): "
if "%msg%"=="" set "msg=update %date% %time%"

git add -A
git commit -m "%msg%"

echo.
echo 正在上传到 GitHub...
git push
if errorlevel 1 (
    echo.
    echo [错误] 上传失败，请检查网络或 SSH 配置。
) else (
    echo.
    echo [成功] 代码已保存到 GitHub！
)

echo.
pause
