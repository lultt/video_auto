@echo off
REM Build cpp_pipeline using Visual Studio 2022 BuildTools
REM Save as build.bat, run from cpp_pipeline/ directory

set CMAKE="C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe"

echo === cmake configure ===
%CMAKE% -B build -G "Visual Studio 17 2022" -A x64

echo.
echo === cmake build (Release) ===
%CMAKE% --build build --config Release

echo.
echo === done ===
echo Binary: build\Release\cpp_pipeline.exe
echo.
echo === quick test ===
build\Release\cpp_pipeline.exe
