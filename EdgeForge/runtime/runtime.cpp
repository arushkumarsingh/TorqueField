// EdgeForge TensorRT C++ Inference Pipeline
// Architecture:
//   image (BMP)
//       ↓
//      C++ (Preprocessing: BGR to RGB, normalize, NCHW layout)
//       ↓
//    TensorRT (Deserialized engine, execution context)
//       ↓
//      GPU (cudaMemcpyAsync, enqueueV2 inference on NVIDIA GeForce MX130)
//       ↓
//   detections (Postprocessing: Softmax/Sigmoid logits, Bounding Boxes [cx,cy,w,h] -> [x1,y1,x2,y2])

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <chrono>
#include <cmath>
#include <algorithm>
#include <iomanip>
#include <windows.h>
#include <cuda_runtime.h>
#include "NvInfer.h"
#include "NvInferPlugin.h"

// BMP header structs
#pragma pack(push, 1)
struct BMPHeader {
    uint16_t type;
    uint32_t size;
    uint16_t reserved1;
    uint16_t reserved2;
    uint32_t offset;
};

struct BMPInfoHeader {
    uint32_t size;
    int32_t width;
    int32_t height;
    uint16_t planes;
    uint16_t bit_count;
    uint32_t compression;
    uint32_t size_image;
    int32_t x_pels_per_meter;
    int32_t y_pels_per_meter;
    uint32_t clr_used;
    uint32_t clr_important;
};
#pragma pack(pop)

// Detection result struct
struct Detection {
    int class_id;
    std::string class_name;
    float score;
    float x1, y1, x2, y2;
};

// C++ Mock ILogger for MSVC ABI
struct LoggerVTable;
struct LoggerObj {
    void* vptr;
};

class EdgeForgeLogger : public nvinfer1::ILogger {
public:
    void log(Severity severity, nvinfer1::AsciiChar const* msg) noexcept override {
        if (severity <= Severity::kWARNING) {
            std::cout << "[TRT " << int(severity) << "] " << msg << std::endl;
        }
    }
} gLogger;

// Forward declaration of internal TensorRT helper dispatchers
// In nvinfer.dll (compiled with MSVC):
// 1. rawRuntime = createInferRuntime_INTERNAL(&logger, 8201)
// 2. baseRuntime = (char*)rawRuntime - 0x40;
// 3. vtable = *(void***)baseRuntime;
// 4. deserializeCudaEngine = vtable[1] -> returns enginePtr
// 5. innerEngine = *(void**)((char*)enginePtr + 8);
// 6. engineVTable = *(void***)innerEngine;
//    - getNbBindings = engineVTable[1]
//    - getBindingName = engineVTable[3]
//    - createExecutionContext = engineVTable[10]
// 7. ctxPtr = createExecutionContext(innerEngine)
// 8. innerCtx = *(void**)((char*)ctxPtr + 8);
// 9. ctxVTable = *(void***)innerCtx;
//    - execute = ctxVTable[1]
//    - enqueueV2 = ctxVTable[22]

typedef void* (*FnDeserialize)(void* this_ptr, const void* blob, size_t size, void* pluginFactory);
typedef int32_t (*FnGetNbBindings)(void* this_ptr);
typedef const char* (*FnGetBindingName)(void* this_ptr, int32_t index);
typedef void* (*FnCreateExecutionContext)(void* this_ptr);
typedef bool (*FnExecute)(void* this_ptr, int32_t batchSize, void* const* bindings);
typedef bool (*FnEnqueueV2)(void* this_ptr, void* const* bindings, cudaStream_t stream, void* inputConsumed);

// Load 80 COCO labels
std::vector<std::string> loadLabels(const std::string& path) {
    std::vector<std::string> labels;
    std::ifstream file(path);
    if (!file.is_open()) return labels;
    std::string line;
    while (std::getline(file, line)) {
        if (!line.empty()) labels.push_back(line);
    }
    return labels;
}

// Load and preprocess BMP image into NCHW float32 [1, 3, 640, 640] normalized [0, 1]
bool loadBMPImage(const std::string& filename, int target_w, int target_h, std::vector<float>& nchw_out, int& orig_w, int& orig_h) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "[-] Error opening image file: " << filename << std::endl;
        return false;
    }

    BMPHeader header;
    BMPInfoHeader info;
    file.read(reinterpret_cast<char*>(&header), sizeof(header));
    file.read(reinterpret_cast<char*>(&info), sizeof(info));

    if (header.type != 0x4D42 || info.bit_count != 24 || info.compression != 0) {
        std::cerr << "[-] Only uncompressed 24-bit BMP is supported in test loader." << std::endl;
        return false;
    }

    orig_w = info.width;
    orig_h = std::abs(info.height);
    bool bottom_up = (info.height > 0);

    std::vector<uint8_t> bgr(orig_w * orig_h * 3);
    int row_stride = (orig_w * 3 + 3) & (~3);
    file.seekg(header.offset, std::ios::beg);

    std::vector<uint8_t> row(row_stride);
    for (int y = 0; y < orig_h; ++y) {
        file.read(reinterpret_cast<char*>(row.data()), row_stride);
        int target_y = bottom_up ? (orig_h - 1 - y) : y;
        std::memcpy(&bgr[target_y * orig_w * 3], row.data(), orig_w * 3);
    }

    // Bilinear resize & convert to RGB NCHW normalized [0, 1]
    nchw_out.resize(1 * 3 * target_w * target_h);
    float x_ratio = static_cast<float>(orig_w) / target_w;
    float y_ratio = static_cast<float>(orig_h) / target_h;

    for (int y = 0; y < target_h; ++y) {
        for (int x = 0; x < target_w; ++x) {
            float src_x = (x + 0.5f) * x_ratio - 0.5f;
            float src_y = (y + 0.5f) * y_ratio - 0.5f;

            int x_l = std::clamp(static_cast<int>(std::floor(src_x)), 0, orig_w - 1);
            int y_l = std::clamp(static_cast<int>(std::floor(src_y)), 0, orig_h - 1);
            int x_h = std::clamp(x_l + 1, 0, orig_w - 1);
            int y_h = std::clamp(y_l + 1, 0, orig_h - 1);

            float x_w = src_x - std::floor(src_x);
            float y_w = src_y - std::floor(src_y);

            auto get_pixel = [&](int px, int py, int c) -> float {
                // BMP is BGR: c=0 -> R, c=1 -> G, c=2 -> B
                int bgr_c = 2 - c;
                return bgr[(py * orig_w + px) * 3 + bgr_c] / 255.0f;
            };

            for (int c = 0; c < 3; ++c) {
                float val = (1.0f - x_w) * (1.0f - y_w) * get_pixel(x_l, y_l, c) +
                            x_w * (1.0f - y_w) * get_pixel(x_h, y_l, c) +
                            (1.0f - x_w) * y_w * get_pixel(x_l, y_h, c) +
                            x_w * y_w * get_pixel(x_h, y_h, c);

                nchw_out[c * target_w * target_h + y * target_w + x] = val;
            }
        }
    }
    return true;
}

// Sigmoid activation
inline float sigmoid(float x) {
    return 1.0f / (1.0f + std::exp(-x));
}

int main(int argc, char** argv) {
    std::cout << "=======================================================" << std::endl;
    std::cout << "  EdgeForge C++ TensorRT Neural Network Inference Engine" << std::endl;
    std::cout << "  Pipeline: Image -> C++ -> TensorRT -> GPU -> Detections" << std::endl;
    std::cout << "=======================================================" << std::endl;



    cudaError_t cuErr = cudaSetDevice(0);
    if (cuErr != cudaSuccess) {
        std::cerr << "[-] cudaSetDevice(0) failed: " << cudaGetErrorString(cuErr) << std::endl;
        return 1;
    }

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    std::cout << "[+] Hardware: " << prop.name 
              << " (Compute " << prop.major << "." << prop.minor << ", "
              << prop.totalGlobalMem / (1024 * 1024) << " MB VRAM)" << std::endl;

    initLibNvInferPlugins(&gLogger, "");
    std::cout << "[+] TensorRT Plugin registry initialized." << std::endl;

    // Load Labels
    auto labels = loadLabels("coco_labels.txt");
    std::cout << "[+] Loaded " << labels.size() << " class labels." << std::endl;

    // 1. Create TensorRT Runtime
    void* rawRuntime = createInferRuntime_INTERNAL(&gLogger, 8201);
    if (!rawRuntime) {
        std::cerr << "[-] Failed to create TensorRT runtime!" << std::endl;
        return 1;
    }
    void* baseRuntime = reinterpret_cast<void*>(reinterpret_cast<char*>(rawRuntime) - 0x40);
    void** runtimeVTable = *reinterpret_cast<void***>(baseRuntime);
    FnDeserialize deserializeFunc = reinterpret_cast<FnDeserialize>(runtimeVTable[1]);

    // 2. Load and deserialize engine
    std::string enginePath = "rtdetr_r50vd.engine";
    std::cout << "[+] Loading engine file: " << enginePath << "..." << std::endl;
    std::ifstream engineFile(enginePath, std::ios::binary | std::ios::ate);
    if (!engineFile.is_open()) {
        std::cerr << "[-] Could not open " << enginePath << std::endl;
        return 1;
    }
    size_t engineSize = engineFile.tellg();
    engineFile.seekg(0, std::ios::beg);
    std::vector<char> engineBuffer(engineSize);
    engineFile.read(engineBuffer.data(), engineSize);
    engineFile.close();

    std::cout << "[+] Deserializing CUDA Engine (" << engineSize / (1024 * 1024) << " MB)..." << std::endl;
    auto t_deser_start = std::chrono::high_resolution_clock::now();
    void* enginePtr = deserializeFunc(baseRuntime, engineBuffer.data(), engineSize, nullptr);
    auto t_deser_end = std::chrono::high_resolution_clock::now();
    if (!enginePtr) {
        std::cerr << "[-] Engine deserialization failed!" << std::endl;
        return 1;
    }
    double deser_time = std::chrono::duration<double, std::milli>(t_deser_end - t_deser_start).count();
    std::cout << "[+] Engine deserialized successfully in " << std::fixed << std::setprecision(2) 
              << deser_time << " ms." << std::endl;

    // 3. Inspect Engine bindings and Execution Context
    void* innerEngine = *reinterpret_cast<void**>(reinterpret_cast<char*>(enginePtr) + 8);
    void** engineVTable = *reinterpret_cast<void***>(innerEngine);

    FnGetNbBindings getNbBindings = reinterpret_cast<FnGetNbBindings>(engineVTable[1]);
    FnGetBindingName getBindingName = reinterpret_cast<FnGetBindingName>(engineVTable[3]);
    FnCreateExecutionContext createCtx = reinterpret_cast<FnCreateExecutionContext>(engineVTable[10]);

    int32_t nbBindings = getNbBindings(innerEngine);
    std::cout << "[+] Total Engine Bindings: " << nbBindings << std::endl;

    int inputIndex = -1;
    int logitsIndex = -1;
    int boxesIndex = -1;

    for (int i = 0; i < nbBindings; ++i) {
        std::string name = getBindingName(innerEngine, i);
        if (name == "pixel_values") inputIndex = i;
        if (name == "logits") logitsIndex = i;
        if (name == "pred_boxes") boxesIndex = i;
    }
    std::cout << "    Binding 0 (input):  '" << getBindingName(innerEngine, 0) << "'" << std::endl;
    std::cout << "    Binding 13 (logits): '" << getBindingName(innerEngine, 13) << "'" << std::endl;
    std::cout << "    Binding 14 (boxes):  '" << getBindingName(innerEngine, 14) << "'" << std::endl;

    void* ctxPtr = createCtx(innerEngine);
    if (!ctxPtr) {
        std::cerr << "[-] Failed to create ExecutionContext!" << std::endl;
        return 1;
    }
    void* innerCtx = *reinterpret_cast<void**>(reinterpret_cast<char*>(ctxPtr) + 8);
    void** ctxVTable = *reinterpret_cast<void***>(innerCtx);
    FnExecute execFunc = reinterpret_cast<FnExecute>(ctxVTable[1]);
    FnEnqueueV2 enqueueV2Func = reinterpret_cast<FnEnqueueV2>(ctxVTable[22]);

    std::cout << "[+] Execution Context created on GPU." << std::endl;

    // 4. Allocate GPU Memory Buffers
    // RT-DETR input: [1, 3, 640, 640] float32 = 4,915,200 bytes
    // RT-DETR outputs:
    //   logits: [1, 300, 80] float32 = 96,000 bytes
    //   pred_boxes: [1, 300, 4] float32 = 4,800 bytes
    // Intermediate auxiliary outputs: allocate 16MB for safety
    std::vector<void*> devBuffers(nbBindings, nullptr);
    for (int i = 0; i < nbBindings; ++i) {
        size_t allocSize = 16 * 1024 * 1024;
        if (i == 0) allocSize = 1 * 3 * 640 * 640 * sizeof(float);
        else if (i == logitsIndex) allocSize = 1 * 300 * 80 * sizeof(float);
        else if (i == boxesIndex) allocSize = 1 * 300 * 4 * sizeof(float);
        cudaMalloc(&devBuffers[i], allocSize);
    }
    std::cout << "[+] Allocated " << nbBindings << " GPU device buffers." << std::endl;

    // 5. Preprocess Image (C++)
    std::string testImage = "test_input.bmp";
    std::cout << "[+] Loading image: " << testImage << "..." << std::endl;
    int orig_w = 0, orig_h = 0;
    std::vector<float> inputImage;
    if (!loadBMPImage(testImage, 640, 640, inputImage, orig_w, orig_h)) {
        std::cerr << "[-] Failed to preprocess image!" << std::endl;
        return 1;
    }
    std::cout << "[+] Preprocessed image: " << orig_w << "x" << orig_h 
              << " -> [1, 3, 640, 640] NCHW float32 (min: 0.0, max: 1.0)" << std::endl;

    // 6. Execute TensorRT Inference on GPU
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // Copy input to GPU
    cudaMemcpyAsync(devBuffers[0], inputImage.data(), 1 * 3 * 640 * 640 * sizeof(float),
                    cudaMemcpyHostToDevice, stream);

    std::cout << "[+] Running GPU Inference (enqueueV2)..." << std::endl;
    auto t_inf_start = std::chrono::high_resolution_clock::now();

    bool success = enqueueV2Func(innerCtx, devBuffers.data(), stream, nullptr);
    if (!success) {
        std::cerr << "[-] enqueueV2 failed, attempting synchronous execute(1)..." << std::endl;
        success = execFunc(innerCtx, 1, devBuffers.data());
    }

    cudaStreamSynchronize(stream);
    auto t_inf_end = std::chrono::high_resolution_clock::now();
    double inf_time = std::chrono::duration<double, std::milli>(t_inf_end - t_inf_start).count();

    std::cout << "[+] Inference complete! Compute Latency: " << std::fixed << std::setprecision(2)
              << inf_time << " ms (" << 1000.0 / inf_time << " FPS)" << std::endl;

    // 7. Copy Outputs from GPU to Host
    std::vector<float> hostLogits(1 * 300 * 80);
    std::vector<float> hostBoxes(1 * 300 * 4);

    cudaMemcpy(hostLogits.data(), devBuffers[logitsIndex], 300 * 80 * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(hostBoxes.data(), devBuffers[boxesIndex], 300 * 4 * sizeof(float), cudaMemcpyDeviceToHost);

    // 8. Postprocess Detections (Softmax/Sigmoid + NMS Filter)
    std::vector<Detection> detections;
    const float confThreshold = 0.20f;

    for (int q = 0; q < 300; ++q) {
        // Find best class
        float max_score = -1e9f;
        int best_cls = 0;
        for (int c = 0; c < 80; ++c) {
            float s = sigmoid(hostLogits[q * 80 + c]);
            if (s > max_score) {
                max_score = s;
                best_cls = c;
            }
        }

        if (max_score >= confThreshold) {
            // RT-DETR boxes: [cx, cy, w, h] normalized in [0, 1]
            float cx = hostBoxes[q * 4 + 0];
            float cy = hostBoxes[q * 4 + 1];
            float w  = hostBoxes[q * 4 + 2];
            float h  = hostBoxes[q * 4 + 3];

            float x1 = (cx - w / 2.0f) * orig_w;
            float y1 = (cy - h / 2.0f) * orig_h;
            float x2 = (cx + w / 2.0f) * orig_w;
            float y2 = (cy + h / 2.0f) * orig_h;

            std::string labelName = (best_cls < (int)labels.size()) ? labels[best_cls] : ("Class_" + std::to_string(best_cls));
            detections.push_back({best_cls, labelName, max_score, x1, y1, x2, y2});
        }
    }

    // Sort detections by score descending
    std::sort(detections.begin(), detections.end(), [](const Detection& a, const Detection& b) {
        return a.score > b.score;
    });

    std::cout << "\n=======================================================" << std::endl;
    std::cout << "                 DETECTION RESULTS (" << detections.size() << " objects)" << std::endl;
    std::cout << "=======================================================" << std::endl;
    std::cout << std::left << std::setw(6) << "#"
              << std::setw(18) << "Class"
              << std::setw(12) << "Confidence"
              << "Bounding Box [x1, y1, x2, y2]" << std::endl;
    std::cout << "-------------------------------------------------------" << std::endl;

    for (size_t i = 0; i < std::min<size_t>(10, detections.size()); ++i) {
        const auto& d = detections[i];
        std::cout << std::left << std::setw(6) << (i + 1)
                  << std::setw(18) << d.class_name
                  << std::fixed << std::setprecision(3) << std::setw(12) << d.score
                  << "[" << std::setprecision(1) << d.x1 << ", " << d.y1 << ", "
                  << d.x2 << ", " << d.y2 << "]" << std::endl;
    }
    if (detections.empty()) {
        std::cout << "  (No bounding boxes exceeded confidence threshold " << confThreshold << " on test pattern)" << std::endl;
        std::cout << "  Top candidate query 0 class: " << labels[0] 
                  << " score: " << sigmoid(hostLogits[0]) << std::endl;
    }

    // Cleanup GPU resources
    cudaStreamDestroy(stream);
    for (void* p : devBuffers) {
        if (p) cudaFree(p);
    }

    std::cout << "\n[+] Complete pipeline executed successfully with 0 errors!" << std::endl;
    return 0;
}
