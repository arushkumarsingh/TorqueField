import os
os.environ["HF_HOME"] = "./hf_cache"

import torch
from transformers import RTDetrForObjectDetection

print("Loading model for export...")
model = RTDetrForObjectDetection.from_pretrained("PekingU/rtdetr_r50vd")
model.eval() # Set the model to inference mode

# RT-DETR expects an image of shape [batch_size, channels, height, width]
batch_size = 1
channels = 3
height = 640
width = 640

print(f"Creating dummy input of shape ({batch_size}, {channels}, {height}, {width})...")
dummy_input = torch.randn(batch_size, channels, height, width)

onnx_file_path = "rtdetr_r50vd.onnx"

print(f"Exporting to {onnx_file_path}...")
torch.onnx.export(
    model,                      
    dummy_input,                
    onnx_file_path,             
    opset_version=16,           
    input_names=["pixel_values"],         
    output_names=["logits", "pred_boxes"],
    dynamic_axes={              
        "pixel_values": {0: "batch_size"},
        "logits": {0: "batch_size"},
        "pred_boxes": {0: "batch_size"}
    }
)

print("Export complete! You can now use this ONNX file with TensorRT.")
