import os
os.environ["HF_HOME"] = "./hf_cache"

import torch
import torch.nn as nn
from transformers import RTDetrForObjectDetection
import transformers.models.rt_detr.modeling_rt_detr as rt_detr_module

def manual_grid_sample(value_l, grid):
    """
    Decomposed bilinear grid_sample replacement compatible with TensorRT 8.2.
    value_l: [N, C, H, W]
    grid: [N, Q, P, 2] in [-1, 1], align_corners=False, padding_mode=zeros
    """
    N, C, H, W = value_l.shape
    _, Q, P, _ = grid.shape

    # Un-normalize coordinates
    x = ((grid[..., 0] + 1.0) * float(W) - 1.0) * 0.5
    y = ((grid[..., 1] + 1.0) * float(H) - 1.0) * 0.5

    x0 = torch.floor(x).long()
    y0 = torch.floor(y).long()
    x1 = x0 + 1
    y1 = y0 + 1

    wa = (x1.to(value_l.dtype) - x) * (y1.to(value_l.dtype) - y)
    wb = (x1.to(value_l.dtype) - x) * (y - y0.to(value_l.dtype))
    wc = (x - x0.to(value_l.dtype)) * (y1.to(value_l.dtype) - y)
    wd = (x - x0.to(value_l.dtype)) * (y - y0.to(value_l.dtype))

    # Zero-padding masks
    mask00 = ((x0 >= 0) & (x0 < W) & (y0 >= 0) & (y0 < H)).unsqueeze(1)
    mask01 = ((x0 >= 0) & (x0 < W) & (y1 >= 0) & (y1 < H)).unsqueeze(1)
    mask10 = ((x1 >= 0) & (x1 < W) & (y0 >= 0) & (y0 < H)).unsqueeze(1)
    mask11 = ((x1 >= 0) & (x1 < W) & (y1 >= 0) & (y1 < H)).unsqueeze(1)

    # Clamp coordinates for safe indexing
    x0_c = torch.clamp(x0, 0, W - 1)
    x1_c = torch.clamp(x1, 0, W - 1)
    y0_c = torch.clamp(y0, 0, H - 1)
    y1_c = torch.clamp(y1, 0, H - 1)

    # Flatten spatial dims to gather
    idx00 = (y0_c * W + x0_c).reshape(N, 1, Q * P).expand(N, C, Q * P)
    idx01 = (y1_c * W + x0_c).reshape(N, 1, Q * P).expand(N, C, Q * P)
    idx10 = (y0_c * W + x1_c).reshape(N, 1, Q * P).expand(N, C, Q * P)
    idx11 = (y1_c * W + x1_c).reshape(N, 1, Q * P).expand(N, C, Q * P)

    val_flat = value_l.reshape(N, C, H * W)
    val00 = torch.gather(val_flat, 2, idx00).reshape(N, C, Q, P) * mask00.to(value_l.dtype)
    val01 = torch.gather(val_flat, 2, idx01).reshape(N, C, Q, P) * mask01.to(value_l.dtype)
    val10 = torch.gather(val_flat, 2, idx10).reshape(N, C, Q, P) * mask10.to(value_l.dtype)
    val11 = torch.gather(val_flat, 2, idx11).reshape(N, C, Q, P) * mask11.to(value_l.dtype)

    return (
        val00 * wa.unsqueeze(1) +
        val01 * wb.unsqueeze(1) +
        val10 * wc.unsqueeze(1) +
        val11 * wd.unsqueeze(1)
    )

def patched_msda_forward(
    self,
    value: torch.Tensor,
    value_spatial_shapes: torch.Tensor,
    value_spatial_shapes_list: list,
    level_start_index: torch.Tensor,
    sampling_locations: torch.Tensor,
    attention_weights: torch.Tensor,
    im2col_step: int,
):
    batch_size, _, num_heads, hidden_dim = value.shape
    _, num_queries, num_heads, num_levels, num_points, _ = sampling_locations.shape
    value_list = value.split([height * width for height, width in value_spatial_shapes_list], dim=1)
    sampling_grids = 2 * sampling_locations - 1
    sampling_value_list = []
    for level_id, (height, width) in enumerate(value_spatial_shapes_list):
        value_l_ = (
            value_list[level_id]
            .flatten(2)
            .transpose(1, 2)
            .reshape(batch_size * num_heads, hidden_dim, height, width)
        )
        sampling_grid_l_ = sampling_grids[:, :, :, level_id].transpose(1, 2).flatten(0, 1)
        sampling_value_l_ = manual_grid_sample(value_l_, sampling_grid_l_)
        sampling_value_list.append(sampling_value_l_)

    attention_weights = attention_weights.transpose(1, 2).reshape(
        batch_size * num_heads, 1, num_queries, num_levels * num_points
    )
    output = (
        (torch.stack(sampling_value_list, dim=-2).flatten(-2) * attention_weights)
        .sum(-1)
        .view(batch_size, num_heads * hidden_dim, num_queries)
    )
    return output.transpose(1, 2).contiguous()

# Patch the MultiScaleDeformableAttention forward pass
rt_detr_module.MultiScaleDeformableAttention.forward = patched_msda_forward

print("Loading model for export...")
model = RTDetrForObjectDetection.from_pretrained("PekingU/rtdetr_r50vd")
model.eval()

batch_size = 1
channels = 3
height = 640
width = 640

print(f"Creating dummy input of shape ({batch_size}, {channels}, {height}, {width})...")
dummy_input = torch.randn(batch_size, channels, height, width)

onnx_file_path = "rtdetr_r50vd.onnx"

print(f"Exporting to {onnx_file_path} (opset 16, legacy TorchScript exporter)...")
torch.onnx.export(
    model,                      
    dummy_input,                
    onnx_file_path,             
    opset_version=16,
    dynamo=False,
    input_names=["pixel_values"],         
    output_names=["logits", "pred_boxes"]
)

print("Export complete! ONNX is fully decomposed for TensorRT 8.2.")
