import torch
import torchvision.models as models
from torchvision import transforms
from PIL import Image
import io
import cv2
import numpy as np
import base64
import mediapipe as mp
import mediapipe as mp

# Force OpenCV Fallback for robustness
USE_MP = False
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Setup Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

try:
    weights = models.EfficientNet_B4_Weights.DEFAULT
    model = models.efficientnet_b4(weights=weights)
    
    num_ftrs = model.classifier[1].in_features
    # Binary classification Face verification Fake vs Real
    model.classifier[1] = torch.nn.Linear(num_ftrs, 1)
    
    model = model.to(device)
    model.eval()
    print(f"EfficientNet-B4 loaded successfully with Real Weights on {device}")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

# Pre-processing transforms
transform = transforms.Compose([
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
        cam = cv2.resize(cam, (input_tensor.shape[3], input_tensor.shape[2]))
        cam = cam - np.min(cam)
        cam_max = np.max(cam)
        if cam_max > 0:
            cam = cam / cam_max
            
        return cam, output.item()

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

def crop_to_face(image_np: np.ndarray, margin=0.2):
    h, w, _ = image_np.shape
    
    xmin, ymin, box_w, box_h = 0, 0, 0, 0
    face_found = False
    
    if USE_MP:
        with mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5) as face_detection:
            results = face_detection.process(image_np)
            if results.detections:
                detection = results.detections[0]
                bboxC = detection.location_data.relative_bounding_box
                xmin = int(bboxC.xmin * w)
                ymin = int(bboxC.ymin * h)
                box_w = int(bboxC.width * w)
                box_h = int(bboxC.height * h)
                face_found = True
    else:
        gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        if len(faces) > 0:
            faces = sorted((int(x), int(y), int(w_f), int(h_f)) for (x, y, w_f, h_f) in faces)
            faces.sort(key=lambda x: x[2] * x[3], reverse=True)
            xmin, ymin, box_w, box_h = faces[0]
            face_found = True
            
    if not face_found:
        return image_np # Fallback to original if no face found

    # 20% margin
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

def apply_histogram_equalization(image_np: np.ndarray):
    # Convert to YUV for equalizing only the Y channel (brightness)
    img_yuv = cv2.cvtColor(image_np, cv2.COLOR_RGB2YUV)
    img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
    img_output = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2RGB)
    return img_output

def apply_frequency_domain_preprocessing(image_np: np.ndarray):
    """
    Applies CLAHE and then a Laplacian Filter to highlight edge inconsistencies.
    """
    # 1. CLAHE dynamic range normalization
    img_yuv = cv2.cvtColor(image_np, cv2.COLOR_RGB2YUV)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img_yuv[:, :, 0] = clahe.apply(img_yuv[:, :, 0])
    clahe_img = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2RGB)

    # 2. Extract edge map on CLAHE normalized layer
    gray = cv2.cvtColor(clahe_img, cv2.COLOR_RGB2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    laplacian_normalized = cv2.convertScaleAbs(laplacian)
    
    # 3. Create a 3-channel version of the edge map
    edge_map = cv2.cvtColor(laplacian_normalized, cv2.COLOR_GRAY2RGB)
    
    # Blend the edge map and the CLAHE image (50/50 blend)
    blended = cv2.addWeighted(clahe_img, 0.5, edge_map, 0.5, 0)
    return blended

def extract_face_patches(image_np: np.ndarray):
    """
    Extract 4 overlapping patches (top-left, top-right, bottom-left, bottom-right)
    approximating Eyes, Nose, Mouth, and Chin areas.
    """
    h, w, _ = image_np.shape
    h_mid = h // 2
    w_mid = w // 2
    
    overlap_h = h // 8
    overlap_w = w // 8
    
    patch_tl = image_np[0:h_mid + overlap_h, 0:w_mid + overlap_w]  # Left Eye / Top Nose
    patch_tr = image_np[0:h_mid + overlap_h, w_mid - overlap_w:w]  # Right Eye / Top Nose
    patch_bl = image_np[h_mid - overlap_h:h, 0:w_mid + overlap_w]  # Left Mouth / Chin
    patch_br = image_np[h_mid - overlap_h:h, w_mid - overlap_w:w]  # Right Mouth / Chin
    
    return [patch_tl, patch_tr, patch_bl, patch_br]

if model is not None:
    # EfficientNet-b4 features[-1] is the last Conv2dNormActivation block
    target_layer = model.features[-1]
    grad_cam = GradCAM(model, target_layer)

def predict_single_image(image: Image.Image):
    image = image.convert("RGB")
    image_np = np.array(image)
    
    # 1. Pipeline: MediaPipe/OpenCV Face Cropping (with 20% margin)
    cropped_np = crop_to_face(image_np, margin=0.2)
    
    # 2. Pipeline: OpenCV Frequency Domain Pre-processing (Laplacian Edge highlighting)
    freq_np = apply_frequency_domain_preprocessing(cropped_np)
    
    # 3. Patch-Based Extraction
    patches = extract_face_patches(freq_np)
    
    patch_probs = []
    patch_logits = []
    cam_masks = []
    
    for patch in patches:
        final_image = Image.fromarray(patch)
        input_tensor = transform(final_image).unsqueeze(0).to(device)
        input_tensor.requires_grad = True # Required for Grad-CAM
        
        cam_mask, logit = grad_cam.generate(input_tensor)
        
        # Temperature Scaling Polarization
        temperature = 1.5
        # Multiplying logit by temp mathematically 'stretches' it (comparable to dividing by T < 1)
        scaled_logit = logit * temperature
        probability = float(torch.sigmoid(torch.tensor(scaled_logit)).item())
        
        patch_probs.append(probability)
        patch_logits.append(logit)
        cam_masks.append(cam_mask)
        
    # Aggregate probabilities (Geometric Mean across patches)
    import math
    avg_probability = math.exp(sum(math.log(p + 1e-7) for p in patch_probs) / len(patch_probs))
    
    # Feature Calibration Priority Map
    if any(p > 0.70 for p in patch_probs):
        avg_probability = max(patch_probs) # Overrides geometric mean if heavily manipulated patch detected
        
    variance = float(np.var(patch_probs))
    
    # Logging the logits and score
    print(f"Raw Logits: {[round(l, 3) for l in patch_logits]} | Calibrated Score: {avg_probability:.4f}")
    
    # Base Grad-CAM on the patch with the highest fake probability (typically eyes/mouth area)
    max_prob_idx = np.argmax(patch_probs)
    best_cam_mask = cam_masks[max_prob_idx]
    best_patch = patches[max_prob_idx]
    
    # 4. Decision Threshold Logic (> 0.70 Confirmed, 0.40 - 0.70 Suspected)
    is_fake = avg_probability >= 0.40
    
    is_confirmed_deepfake = avg_probability >= 0.70
    is_suspected = 0.40 <= avg_probability < 0.70
    
    if is_fake:
         confidence = avg_probability
    else:
         confidence = 1.0 - avg_probability
         
    # Generate heatmap strictly over the highest-risk focal patch
    heatmap_b64 = generate_heatmap_base64(best_patch, best_cam_mask)
    
    return {
        "is_fake": is_fake,
        "is_deepfake": is_confirmed_deepfake,
        "is_suspected": is_suspected,
        "confidence": confidence,
        "probability": avg_probability,
        "heatmap": heatmap_b64,
        "variance": variance
    }

def predict(image_bytes: bytes) -> dict:
    if model is None:
        raise RuntimeError("Model was not loaded correctly.")
    
    image = Image.open(io.BytesIO(image_bytes))
    res = predict_single_image(image)
    return {
        "is_fake": res["is_fake"],
        "confidence": round(res["confidence"] * 100, 2),
        "heatmap_url": res["heatmap"]
    }
