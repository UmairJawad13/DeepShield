import os
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from PIL import Image
import io
import cv2
import numpy as np
import base64

# Force OpenCV Fallback for robustness (Only for Bounding Box)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Setup Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 3. Architecture & Weights Synchronization
model_path = "DeepShield.pth"
architecture_loaded = "Unknown"

try:
    model = models.efficientnet_b4()
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_ftrs, 1) # Force Binary Output
    
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location='cpu'), strict=True)
        architecture_loaded = "B4"
    else:
        model = models.efficientnet_b4(weights=models.EfficientNet_B4_Weights.DEFAULT)
        model.classifier[1] = nn.Linear(num_ftrs, 1)
        architecture_loaded = "B4"
        
    model = model.to(device)
    model.eval()
    print("[SYSTEM] Strict BGR2RGB and Binary Head applied.")
except Exception as e:
    print(f"[SYSTEM] B4 Initialization failed: {e}. Falling back to EfficientNet-B0.")
    model = models.efficientnet_b0()
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_ftrs, 1) # Force Binary Output
    
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location='cpu'), strict=True)
        architecture_loaded = "B0"
    else:
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        model.classifier[1] = nn.Linear(num_ftrs, 1)
        architecture_loaded = "B0"
        
    model = model.to(device)
    model.eval()

# Pure PIL & Torchvision Transform Pipeline
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate(self, input_tensor):
        self.model.zero_grad()
        output = self.model(input_tensor)
        
        target = output[0, 0] # Binary classification output
        target.backward(retain_graph=True)
        
        gradients = self.gradients.data.cpu().numpy()[0]
        activations = self.activations.data.cpu().numpy()[0]
        
        weights = np.mean(gradients, axis=(1, 2))
        
        cam = np.zeros(activations.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * activations[i]
            
        cam = np.maximum(cam, 0)
        # Cam resize using cv2 is fine, it's just the heatmap
        cam = cv2.resize(cam, (input_tensor.shape[3], input_tensor.shape[2]))
        cam = cam - np.min(cam)
        cam_max = np.max(cam)
        if cam_max > 0:
            cam = cam / cam_max
            
        return cam, torch.sigmoid(output).item()

def generate_heatmap_base64(pil_image: Image.Image, cam_mask: np.ndarray) -> str:
    image_np = np.array(pil_image)
    cam_mask_resized = cv2.resize(cam_mask, (image_np.shape[1], image_np.shape[0]))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_mask_resized), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    
    alpha = 0.5
    overlay = cv2.addWeighted(image_np, 1 - alpha, heatmap, alpha, 0)
    
    overlay_img = Image.fromarray(overlay)
    buffered = io.BytesIO()
    overlay_img.save(buffered, format="JPEG", quality=85)
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    
    return f"data:image/jpeg;base64,{img_str}"

def get_face_bbox(image_np: np.ndarray, margin=0.30):
    """
    Use OpenCV purely to grab bounding box coords, not to manipulate the actual image stream.
    """
    h, w, _ = image_np.shape
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    
    if len(faces) == 0:
        return (0, 0, w, h) # Full image
        
    faces = sorted((int(x), int(y), int(w_f), int(h_f)) for (x, y, w_f, h_f) in faces)
    faces.sort(key=lambda x: x[2] * x[3], reverse=True)
    xmin, ymin, box_w, box_h = faces[0]
    
    margin_x = int(box_w * margin)
    margin_y = int(box_h * margin)

    x1 = max(0, xmin - margin_x)
    y1 = max(0, ymin - margin_y)
    x2 = min(w, xmin + box_w + margin_x)
    y2 = min(h, ymin + box_h + margin_y)

    return (x1, y1, x2, y2)

if model is not None:
    # Get last conv layer depending on architecture
    target_layer = model.features[-1]
    grad_cam = GradCAM(model, target_layer)

def predict_single_image(image: Image.Image):
    image = image.convert("RGB")
    
    # 1. Grab coordinates using OpenCV Array purely as a map
    image_np = np.array(image)
    (left, top, right, bottom) = get_face_bbox(image_np, margin=0.30)

    # 2. Native PIL Cropping
    cropped_image = image.crop((left, top, right, bottom))
    
    # 3. Bulletproof PyTorch Pipeline
    input_tensor = preprocess(cropped_image).unsqueeze(0).to(device)
    input_tensor.requires_grad = True # Required for Grad-CAM
    
    cam_mask, probability = grad_cam.generate(input_tensor)
    
    print(f"[SYSTEM] Architecture Loaded: {architecture_loaded} | Raw Score: {probability:.4f}")
    
    threshold = 0.50
    is_fake = probability >= threshold
    
    if is_fake:
         confidence = probability
    else:
         confidence = 1.0 - probability
         
    # Grad-CAM mapped directly to PIL crop resized to standard sizes
    # We pass the natively resized PIL crop so it aligns
    standardized_crop = cropped_image.resize((224, 224))
    heatmap_b64 = generate_heatmap_base64(standardized_crop, cam_mask)
    
    return {
        "is_fake": is_fake,
        "is_deepfake": is_fake,
        "confidence": confidence,
        "probability": probability,
        "heatmap": heatmap_b64,
        "architecture_loaded": architecture_loaded
    }

def predict(image_bytes: bytes) -> dict:
    if model is None:
        raise RuntimeError("Model was not loaded correctly.")
    
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    res = predict_single_image(image)
    return {
        "is_fake": res["is_fake"],
        "confidence": round(res["confidence"] * 100, 2),
        "heatmap_url": res["heatmap"]
    }
