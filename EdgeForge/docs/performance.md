# EdgeForge Performance Report

## Platform Specifications
- **GPU:** NVIDIA GeForce MX130 (Maxwell Architecture, Compute Capability 5.0, 2047 MB VRAM)
- **CUDA Version:** 11.4
- **cuDNN Version:** 8.2.1
- **TensorRT Version:** 8.2.1.8
- **OS:** Windows 10 (64-bit)

## Architecture Pipeline
```
Image -> C++ -> TensorRT -> GPU -> Detections
```

## Inference Benchmark (RT-DETR ResNet-50vd)
- **Input Resolution:** `1 x 3 x 640 x 640` (NCHW, FP32)
- **Output Tensors:**
  - `logits`: `1 x 300 x 80` (300 query vectors across 80 COCO classes)
  - `pred_boxes`: `1 x 300 x 4` (normalized [cx, cy, w, h] coordinates)
- **Engine Deserialization Time:** ~1026 ms
- **GPU Compute Latency (`enqueueV2`):** ~250.93 ms (~4 FPS on MX130)
- **Status:** PASS (0 errors, verified bounding boxes)
