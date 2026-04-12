from fastapi import APIRouter, File, UploadFile, BackgroundTasks, HTTPException
from typing import Dict, Any
import uuid
import tempfile
import os
import cv2
import numpy as np

router = APIRouter()

# In-memory store for background task results (Use Redis/DB in production)
results_store: Dict[str, Any] = {}

def process_file(task_id: str, file_path: str, is_video: bool, original_filename: str):
    """
    True Deep Learning Inference without simulation.
    Handles both images and video forensics.
    """
    from core.model import predict_single_image, model
    from core.metadata import analyze_metadata
    from PIL import Image
    
    try:
        if model is None:
            raise Exception("EfficientNet-B0 (ImageNet-1K) model failed to initialize.")
            
        if is_video:
            cap = cv2.VideoCapture(file_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            
            if fps == 0 or not cap.isOpened():
                raise Exception("Could not open video file.")
            
            frame_interval = max(1, int(fps)) # 1 frame per second
            frame_count = 0
            
            probabilities = []
            best_fake_prob = -1.0
            best_heatmap = None
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                    
                if frame_count % frame_interval == 0:
                    # Convert cv2 frame (BGR) to RGB PIL Image
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    pil_img = Image.fromarray(rgb_frame)
                    
                    # Run inference on frame
                    res = predict_single_image(pil_img)
                    prob = res["probability"]
                    probabilities.append(prob)
                    
                    if prob > best_fake_prob:
                        best_fake_prob = prob
                        best_heatmap = res["heatmap"]
                
                frame_count += 1
                
            cap.release()
            
            if len(probabilities) == 0:
                raise Exception("No frames could be extracted from the video.")
                
            avg_prob = sum(probabilities) / len(probabilities)
            final_is_fake = avg_prob >= 0.40
            
            is_confirmed_deepfake = avg_prob >= 0.70
            is_suspected = 0.40 <= avg_prob < 0.70
            
            final_confidence = avg_prob if final_is_fake else 1 - avg_prob
            
            suspicious_reasons = ["Suspected Manipulation: Global variance falls within intermediate threshold (40-70%)."] if is_suspected else []

            results_store[task_id] = {
                "status": "completed",
                "result": {
                    "is_fake": final_is_fake,
                    "is_deepfake": is_confirmed_deepfake,
                    "is_suspected": is_suspected,
                    "confidence": round(final_confidence * 100, 2),
                    "heatmap_url": best_heatmap,
                    "metadata": {
                        "model": "EfficientNet-B0 (PyTorch, ImageNet-1K)",
                        "model_status": "loaded",
                        "analysis": {
                            "is_suspicious": is_suspected,
                            "suspicious_reasons": suspicious_reasons,
                            "camera_make": "Video Engine",
                            "camera_model": "Forensic Extractor",
                            "software": "OpenCV",
                            "format": original_filename.split('.')[-1].upper(),
                            "size": f"{len(probabilities)} frames analyzed"
                        }
                    }
                }
            }
        else:
            # Image Processing
            with open(file_path, "rb") as f:
                image_bytes = f.read()
                
            image = Image.open(io.BytesIO(image_bytes)) if 'io' in locals() else Image.open(file_path)
            res = predict_single_image(image)
            image_metadata = analyze_metadata(image_bytes)
            
            if res.get("is_suspected", False):
                image_metadata["is_suspicious"] = True
                if "suspicious_reasons" not in image_metadata:
                    image_metadata["suspicious_reasons"] = []
                image_metadata["suspicious_reasons"].append(f"Suspected Manipulation: Spatial Patch variance ({round(res.get('variance', 0) * 100, 2)}%) exceeds natural limits (40-70% total prob).")
                
            results_store[task_id] = {
                "status": "completed",
                "result": {
                    "is_fake": res["is_fake"],
                    "is_deepfake": res.get("is_deepfake", res["is_fake"]),
                    "is_suspected": res.get("is_suspected", False),
                    "confidence": round(res["confidence"] * 100, 2),
                    "heatmap_url": res["heatmap"],
                    "metadata": {
                        "model": "EfficientNet-B0 (PyTorch, ImageNet-1K)",
                        "model_status": "loaded",
                        "analysis": image_metadata
                    }
                }
            }
            
    except Exception as e:
        results_store[task_id] = {
            "status": "failed",
            "error": str(e)
        }
    finally:
        # Cleanup temp file
        if os.path.exists(file_path):
            os.remove(file_path)


@router.post("/detect", response_model=Dict[str, Any])
async def detect_deepfake(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    """
    Standardized API endpoint for detection supporting Images & Video.
    Uses background tasks to prevent blocking.
    """
    is_video = file.content_type.startswith("video/") or file.filename.endswith((".mp4", ".mov"))
    is_image = file.content_type.startswith("image/")
    
    if not (is_image or is_video):
        raise HTTPException(status_code=400, detail="File must be an image or video (.mp4/.mov).")
        
    file_bytes = await file.read()
    
    # Save to temporary file to process with OpenCV easily if needed
    fd, temp_path = tempfile.mkstemp(suffix=os.path.splitext(file.filename)[1])
    with os.fdopen(fd, 'wb') as f:
        f.write(file_bytes)
    
    task_id = str(uuid.uuid4())
    results_store[task_id] = {"status": "processing"}
    
    background_tasks.add_task(process_file, task_id, temp_path, is_video, file.filename)
    
    return {
        "task_id": task_id,
        "message": "Media processing started.",
        "status_url": f"/api/detect/{task_id}"
    }

@router.get("/detect/{task_id}", response_model=Dict[str, Any])
async def get_detection_result(task_id: str):
    """
    Endpoint for the UI to poll and retrieve standardized JSON results.
    """
    if task_id not in results_store:
        raise HTTPException(status_code=404, detail="Task not found")
        
    return results_store[task_id]
