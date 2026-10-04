# EdgeForge

High-performance real-time neural network inference engine written in C++ targeting NVIDIA TensorRT and CUDA GPU acceleration.

## Architecture

```
image (BMP/Camera frame)
    ↓
C++ (Preprocessing: bilinear resize, BGR->RGB, normalization to [0, 1], NCHW conversion)
    ↓
TensorRT (Deserialized engine, execution context)
    ↓
GPU (Asynchronous CUDA streams, enqueueV2 kernel execution on NVIDIA GPU)
    ↓
detections (Postprocessing: Sigmoid classification scores, bounding box un-normalization)
```

## Directory Structure

```
EdgeForge/
├── README.md
├── runtime/
│   ├── runtime.cpp         # Main C++ TensorRT inference pipeline executable
│   ├── pipeline.cpp        # End-to-end stream inference orchestration
│   ├── queue.cpp           # Lock-free buffer queuing
│   └── tracker.cpp         # Multi-object tracking module
├── cuda/
│   ├── preprocess.cu       # GPU CUDA image warp & normalize kernels
│   └── postprocess.cu      # GPU CUDA bounding box decoding & NMS kernels
├── tensorrt/
│   └── engine/             # Serialized TensorRT model engines (.engine)
├── benchmarks/
│   ├── streams/            # Multi-stream concurrent inference tests
│   ├── graphs/             # CUDA Graph execution benchmarks
│   ├── batching/           # Dynamic batching evaluations
│   └── realtime/           # End-to-end camera latency benchmarks
├── profiling/              # Nsight Systems & layer execution metrics
├── results/
│   ├── latency.png
│   ├── throughput.png
│   └── benchmark.csv
└── docs/
    └── performance.md      # Performance benchmarking analysis
```

## Running the Inference Pipeline

```powershell
$env:PATH = "D:\dev-cache\TensorRT-8.2.1.8.Windows10.x86_64.cuda-11.4.cudnn8.2\TensorRT-8.2.1.8\lib;C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.4\bin;D:\dev-cache\llvm-mingw\llvm-mingw-20240619-ucrt-x86_64\bin;" + $env:PATH
.\EdgeForge\runtime\runtime.exe
```
