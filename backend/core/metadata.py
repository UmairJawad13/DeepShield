from exif import Image as ExifImage
from PIL import Image as PilImage
import io
from typing import Dict, Any

def analyze_metadata(image_bytes: bytes) -> Dict[str, Any]:
    """
    Extracts metadata from the uploaded image and flags indicators of AI generation.
    Looks for tags like 'Software', 'MakerNote', or missing 'Make'/'Model'.
    """
    metadata: Dict[str, Any] = {
        "camera_make": "Unknown",
        "camera_model": "Unknown",
        "software": "None",
        "is_suspicious": False,
        "suspicious_reasons": []
    }
    
    try:
        try:
            exif_img = ExifImage(image_bytes)
            has_exif = exif_img.has_exif
        except Exception:
            has_exif = False
            
        if has_exif:
            metadata["camera_make"] = exif_img.get('make', 'Unknown')
            metadata["camera_model"] = exif_img.get('model', 'Unknown')
            metadata["software"] = exif_img.get('software', 'None')
            
            # Check for suspicious flags
            software_lower = str(metadata["software"]).lower()
            if any(ai_term in software_lower for ai_term in ['midjourney', 'dall-e', 'stable diffusion', 'runway', 'photoshop', 'gimp']):
                metadata["is_suspicious"] = True
                metadata["suspicious_reasons"].append(f"Suspicious Software tag: {metadata['software']}")
                
        else:
            # We omit the blanket "Missing EXIF" rule because it causes massive False Positives 
            # for Original PNGs, Web Screenshots, and social media images that strip data.
            pass
            
        # Pillow fallback/supplement for basic image info
        pil_img = PilImage.open(io.BytesIO(image_bytes))
        metadata["format"] = pil_img.format
        metadata["mode"] = pil_img.mode
        metadata["size"] = f"{pil_img.size[0]}x{pil_img.size[1]}"
        
        # Check standard Pillow info for comments or other indicators
        if 'parameters' in pil_img.info or 'prompt' in pil_img.info or 'workflow' in pil_img.info:
            metadata["is_suspicious"] = True
            metadata["suspicious_reasons"].append("Found AI generation parameters in metadata")
        
        return metadata
    except Exception as e:
        print(f"Metadata extraction error: {e}")
        return metadata

