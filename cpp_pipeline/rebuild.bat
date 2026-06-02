@echo off
cd /d "J:\video_auto\cpp_pipeline\build"
"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\MSBuild\Current\Bin\amd64\MSBuild.exe" cpp_pipeline.vcxproj -p:Configuration=Release -p:Platform=x64 -v:minimal
if %ERRORLEVEL% NEQ 0 (
    echo BUILD FAILED
    exit /b 1
)
echo BUILD SUCCESS
