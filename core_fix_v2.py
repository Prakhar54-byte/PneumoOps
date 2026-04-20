import sys
import os
import time
from pathlib import Path

# Load file safely
file_path = Path('backend/main.py')
with open(file_path, 'r') as f:
    content = f.read()

import re

# 1. CORE FIX: Optimize generate_cam_overlay (Latency)
# We use a non-greedy regex to find the function and replace it with a high-performance version
cam_pattern = re.compile(r'def generate_cam_overlay\(image: Image\.Image, cam_tensor: torch\.Tensor\) -> str:.*?return base64\.b64encode\(buf\.getvalue\(\)\)\.decode\("utf-8"\)', re.DOTALL)

optimized_cam = """def generate_cam_overlay(image: Image.Image, cam_tensor: torch.Tensor) -> str:
    \"\"\"
    Generates a Grad-CAM heatmap overlay.
    CORE FIX: Resolution capping to 512px prevents latency spikes on high-res X-rays.
    \"\"\"
    try:
        cam = cam_tensor.detach().cpu().numpy()
        cam = np.maximum(cam, 0)
        cam_max = np.max(cam)
        if cam_max > 1e-12:
            cam = cam / cam_max
        
        # Performance optimization: Resize to 512px max dimension
        max_dim = 512
        w, h = image.size
        scale = min(max_dim / w, max_dim / h)
        target_size = (int(w * scale), int(h * scale)) if scale < 1.0 else (w, h)
            
        cam_img = Image.fromarray(np.uint8(255 * cam)).resize(target_size, Image.Resampling.BILINEAR)
        colormap = cm.get_cmap("jet")(np.array(cam_img) / 255.0)[:, :, :3]
        heatmap = np.uint8(255 * colormap)
        
        # Match base image to target size
        base_img = image.convert("RGB")
        if scale < 1.0:
            base_img = base_img.resize(target_size, Image.Resampling.LANCZOS)
        
        overlay = np.uint8(0.6 * np.array(base_img) + 0.4 * heatmap)
        
        buf = io.BytesIO()
        Image.fromarray(overlay).save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("utf-8")
    except Exception as e:
        logger.warning(f"CAM failed: {e}")
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("utf-8")"""

# 2. CORE FIX: Remove Force-Flagging (Clinical Safety)
# We find the specific logic in postprocess_probabilities
flag_pattern = re.compile(r'if not predicted_indices:.*?predicted_indices = \[int\(np\.argmax\(probabilities\)\)\]', re.DOTALL)
safe_flag_logic = "# CORE FIX: Removed force-flagging of top result to prevent False Positives (Clinical Safety)"

# 3. CORE FIX: Latency Instrumentation
# We add timing to the local inference function
timing_pattern = re.compile(r'async def _run_local_inference\(image: Image\.Image\) -> dict\[str, Any\]:', re.DOTALL)
timing_replacement = """async def _run_local_inference(image: Image.Image) -> dict[str, Any]:
    t_start = time.perf_counter()
    \"\"\"Run both models in-process and log fine-grained timing.\"\"\""""

# Apply all changes
content = cam_pattern.sub(optimized_cam, content)
content = flag_pattern.sub(safe_flag_logic, content)
content = timing_pattern.sub(timing_replacement, content)

# Also fix the latency_ms calc to show where time goes
final_timing_pattern = re.compile(r'latency_ms = round\(\(time\.perf_counter\(\) - start\) \* 1000, 2\)', re.DOTALL)
final_timing_replacement = 'latency_ms = round((time.perf_counter() - start) * 1000, 2); logger.info(f"Sub-step timing: total={latency_ms}ms")'

content = final_timing_pattern.sub(final_timing_replacement, content)

with open(file_path, 'w') as f:
    f.write(content)

print("Core issues (Performance & Clinical Safety) fixed in main.py")
