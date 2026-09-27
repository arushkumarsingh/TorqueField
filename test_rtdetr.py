import os
os.environ["HF_HOME"] = "./hf_cache"

import torch
import requests
from PIL import Image
from transformers import RTDetrForObjectDetection, RTDetrImageProcessor

# 1. Initialize the Image Processor and the Model
# We are using the ResNet-50 version of RT-DETR here.
model_id = "PekingU/rtdetr_r50vd"

print("Downloading and loading the model...")
processor = RTDetrImageProcessor.from_pretrained(model_id)
model = RTDetrForObjectDetection.from_pretrained(model_id)

# 2. Load a sample test image from the internet
print("Fetching a test image...")
url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)

# 3. Preprocess the image
# This handles resizing and normalizing the image exactly how the model expects
inputs = processor(images=image, return_tensors="pt")

# 4. Run Inference (Forward pass)
print("Running inference...")
with torch.no_grad():
    outputs = model(**inputs)

# 5. Post-process the outputs to get human-readable bounding boxes
# We pass the original image size so the bounding boxes scale correctly
target_sizes = torch.tensor([image.size[::-1]])
results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.5)[0]

# Print out what the model found!
print("\n--- Detections ---")
for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
    box = [round(i, 2) for i in box.tolist()]
    label_name = model.config.id2label[label.item()]
    print(f"Detected {label_name} with confidence {round(score.item(), 3)} at location {box}")
