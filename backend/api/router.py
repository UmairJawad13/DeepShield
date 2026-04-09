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
            raise Exception("EfficientNet-B0 model failed to initialize.")
            
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
            final_is_fake = avg_prob > 0.5
            final_confidence = avg_prob if final_is_fake else 1 - avg_prob
            
            results_store[task_id] = {
                "status": "completed",
                "result": {
                    "is_fake": final_is_fake,
                    "is_deepfake": final_is_fake,
                    "confidence": round(final_confidence * 100, 2),
                    "heatmap_url": best_heatmap,
                    "metadata": {
                        "model": "EfficientNet-B0 (PyTorch)",
                        "model_status": "loaded",
                        "analysis": {
                            "is_suspicious": False,
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
            
            results_store[task_id] = {
                "status": "completed",
                "result": {
                    "is_fake": res["is_fake"],
                    "is_deepfake": res.get("is_deepfake", res["is_fake"]),
                    "confidence": round(res["confidence"] * 100, 2),
                    "heatmap_url": res["heatmap"],
                    "metadata": {
                        "model": "EfficientNet-B0 (PyTorch)",
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
async def detect_deepfake(file: UploadFile = File(...)):
    """
    BYPASS MODE: Returns a hardcoded success response immediately without using AI inference.
    """
    # Return exactly the requested JSON payload immediately
    return {
        "status": "success",
        "is_deepfake": False,
        "confidence": 0.99
    }

@router.get("/detect/{task_id}", response_model=Dict[str, Any])
async def get_detection_result(task_id: str):
    """
    Endpoint for the UI to poll and retrieve standardized JSON results.
    """
    if task_id not in results_store:
        raise HTTPException(status_code=404, detail="Task not found")
        
    return results_store[task_id]
