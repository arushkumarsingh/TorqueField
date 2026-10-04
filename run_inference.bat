@echo off
setlocal
cd /d "%~dp0"

echo =======================================================
echo   Starting EdgeForge TensorRT C++ Inference Engine...
echo =======================================================

:: Set paths to TensorRT, CUDA 11.4, and LLVM-MinGW runtimes
set "PATH=D:\dev-cache\TensorRT-8.2.1.8.Windows10.x86_64.cuda-11.4.cudnn8.2\TensorRT-8.2.1.8\lib;C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.4\bin;D:\dev-cache\llvm-mingw\llvm-mingw-20240619-ucrt-x86_64\bin;%PATH%"

:: Run the compiled binary
EdgeForge\runtime\runtime.exe

pause
