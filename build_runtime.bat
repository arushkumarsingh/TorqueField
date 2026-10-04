@echo off
setlocal
cd /d "%~dp0"

echo =======================================================
echo   Compiling EdgeForge C++ Inference Engine...
echo =======================================================

set "CLANG=D:\dev-cache\llvm-mingw\llvm-mingw-20240619-ucrt-x86_64\bin\clang++.exe"
set "TRT_INC=D:\dev-cache\TensorRT-8.2.1.8.Windows10.x86_64.cuda-11.4.cudnn8.2\TensorRT-8.2.1.8\include"
set "TRT_LIB=D:\dev-cache\TensorRT-8.2.1.8.Windows10.x86_64.cuda-11.4.cudnn8.2\TensorRT-8.2.1.8\lib"
set "CUDA_INC=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.4\include"
set "CUDA_LIB=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.4\lib\x64"

"%CLANG%" -O2 -std=c++17 ^
    -I"%TRT_INC%" -I"%CUDA_INC%" ^
    -L"%TRT_LIB%" -L"%CUDA_LIB%" ^
    EdgeForge\runtime\runtime.cpp ^
    -lnvinfer -lnvinfer_plugin -lcudart ^
    -o EdgeForge\runtime\runtime.exe

if %ERRORLEVEL% EQU 0 (
    echo [OK] Compilation succeeded: EdgeForge\runtime\runtime.exe
) else (
    echo [-] Compilation failed with error code %ERRORLEVEL%
)

pause
