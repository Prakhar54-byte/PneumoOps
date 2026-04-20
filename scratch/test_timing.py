import torch
from PIL import Image
import os
import sys

# Ensure backend can be imported
sys.path.append(os.getcwd())

from backend.main import run_pytorch_inference, PYTORCH_MODEL, TRANSFORM
import logging

logging.basicConfig(level=logging.INFO)

def test_timing():
    if PYTORCH_MODEL is None:
        print("Model not loaded")
        return
    
    # Large image to test resizing optimization
    img = Image.new('RGB', (2000, 2000), color = 'red')
    print("Starting inference...")
    result = run_pytorch_inference(img)
    print(f"Total Latency in Result: {result['latency_ms']} ms")

if __name__ == "__main__":
    test_timing()
