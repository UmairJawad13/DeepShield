import urllib.request
import io
from PIL import Image
import torch
import sys
sys.path.append("c:/Users/user/Documents/UOW/FYP 2/System/DeepShield/backend")
from core.model import predict_single_image

try:
    print("Downloading sample face...")
    url = "https://raw.githubusercontent.com/opencv/opencv/master/samples/data/lena.jpg"
    req = urllib.request.urlopen(url)
    img_bytes = req.read()
    
    print("Loading PIL structure...")
    image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    
    print("Testing pipeline...")
    res = predict_single_image(image)
    print("SUCCESS!")
    print(res)
except Exception as e:
    print("CRASHED!")
    import traceback
    traceback.print_exc()
