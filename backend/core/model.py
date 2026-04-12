import os
import torch
import torchvision.models as models
from torchvision import transforms
from PIL import Image
import io
import cv2
import numpy as np
import base64

# Force OpenCV Fallback for robustness
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Setup Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 3. Architecture & Weights Synchronization
model_path = "DeepShield.pth"
architecture_loaded = "Unknown"

try:
    model = models.efficientnet_b4()
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = torch.nn.Linear(num_ftrs, 1)
    
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location='cpu'), strict=True)
        architecture_loaded = "B4"
    else:
        model = models.efficientnet_b4(weights=models.EfficientNet_B4_Weights.DEFAULT)
        model.classifier[1] = torch.nn.Linear(num_ftrs, 1)
        architecture_loaded = "B4"
        
    model = model.to(device)
    model.eval()
    print("[SYSTEM] Strict BGR2RGB and Binary Head applied.")
except Exception as e:
    print(f"[SYSTEM] B4 Initialization failed: {e}. Falling back to EfficientNet-B0.")
    model = models.efficientnet_b0()
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = torch.nn.Linear(num_ftrs, 1) # Force Binary Output
    
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location='cpu'), strict=True)
        architecture_loaded = "B0"
    else:
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        model.classifier[1] = torch.nn.Linear(num_ftrs, 1)
        architecture_loaded = "B0"
        
    model = model.to(device)
    model.eval()

# Pre-processing transforms
transform = transforms.Compose([
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
        cam = cv2.resize(cam, (input_tensor.shape[3], input_tensor.shape[2]))
        cam = cam - np.min(cam)
        cam_max = np.max(cam)
        if cam_max > 0:
            cam = cam / cam_max
            
        # Standard Sigmoid Activation
        return cam, torch.sigmoid(output).item()

def generate_heatmap_base64(image_np: np.ndarray, cam_mask: np.ndarray) -> str:
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

def crop_to_face(image_np: np.ndarray, margin=0.30):
    h, w, _ = image_np.shape
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    
    if len(faces) == 0:
        return image_np
        
    faces = sorted((int(x), int(y), int(w_f), int(h_f)) for (x, y, w_f, h_f) in faces)
    faces.sort(key=lambda x: x[2] * x[3], reverse=True)
    xmin, ymin, box_w, box_h = faces[0]
    
    margin_x = int(box_w * margin)
    margin_y = int(box_h * margin)

    x1 = max(0, xmin - margin_x)
    y1 = max(0, ymin - margin_y)
    x2 = min(w, xmin + box_w + margin_x)
    y2 = min(h, ymin + box_h + margin_y)

    cropped = image_np[y1:y2, x1:x2]
    if cropped.size == 0:
        return image_np
    return cropped

if model is not None:
    # Get last conv layer depending on architecture
    target_layer = model.features[-1]
    grad_cam = GradCAM(model, target_layer)

def predict_single_image(image_np: np.ndarray):
    """
    image_np: strictly an RGB numpy array decoded from OpenCV bytes
    """
    # 1. Contextual Face Extraction (CRITICAL)
    cropped_np = crop_to_face(image_np, margin=0.30)

    # 2. High-Frequency Preservation (Resizing)
    resized_np = cv2.resize(cropped_np, (224, 224), interpolation=cv2.INTER_LANCZOS4)
    
    # 4. Pure Normalization (applied directly to RGB interpolation)
    input_tensor = transform(resized_np).unsqueeze(0).to(device)
    input_tensor.requires_grad = True # Required for Grad-CAM
    
    cam_mask, probability = grad_cam.generate(input_tensor)
    
    print(f"[SYSTEM] Architecture Loaded: {architecture_loaded} | Raw Score: {probability:.4f}")
    
    threshold = 0.50
    is_fake = probability >= threshold
    
    if is_fake:
         confidence = probability
    else:
         confidence = 1.0 - probability
         
    # Grad-CAM mapped directly to cropped array
    heatmap_b64 = generate_heatmap_base64(resized_np, cam_mask)
    
    return {
        "is_fake": is_fake,
        "is_deepfake": is_fake,
        "confidence": confidence,
        "probability": probability,
        "heatmap": heatmap_b64,
        "architecture": architecture_loaded
    }

def predict(image_bytes: bytes) -> dict:
    if model is None:
        raise RuntimeError("Model was not loaded correctly.")
    
    nparr = np.frombuffer(image_bytes, np.uint8)
    image_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if image_bgr is None:
        raise ValueError("Failed to decode image stream.")
    
    # Enforce BGR to RGB Conversion
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    
    res = predict_single_image(image_rgb)
    return {
        "is_fake": res["is_fake"],
        "confidence": round(res["confidence"] * 100, 2),
        "heatmap_url": res["heatmap"]
    }
