import sys
import time
from pathlib import Path

file_path = Path('backend/main.py')
with open(file_path, 'r') as f:
    lines = f.readlines()

# 1. Optimize generate_cam_overlay (Resolution Capping)
# We find the function and modify the resize logic
def find_func_range(lines, func_name):
    start = -1
    for i, line in enumerate(lines):
        if line.strip().startswith(f'def {func_name}'):
            start = i
            break
    if start == -1: return -1, -1
    
    end = -1
    for i in range(start + 1, len(lines)):
        # Look for the end of the try/except block
        if 'return base64' in lines[i] and (i+1 == len(lines) or not lines[i+1].startswith(' ')):
            # This is a bit heuristic, let's find the 'except' block end
            for j in range(i, len(lines)):
                 if 'return base64' in lines[j] and 'except' in lines[j-5:j]:
                     end = j + 1
                     return start, end
    return start, end

# 2. Fix generate_cam_overlay
cam_start, cam_end = find_func_range(lines, "generate_cam_overlay")
if cam_start != -1:
    optimized_cam = """def generate_cam_overlay(image: Image.Image, cam_tensor: torch.Tensor) -> str:
    \"\"\"
    Generates a Grad-CAM heatmap overlay. 
    Optimized: Caps resolution to 512px to prevent latency spikes on large images.
    \"\"\"
    try:
        # 1. Process CAM tensor
        cam = cam_tensor.detach().cpu().numpy()
        cam = np.maximum(cam, 0)
        cam_max = np.max(cam)
        if cam_max > 1e-12:
            cam = cam / cam_max
        
        # 2. Downscale overlay target for performance
        # High-res X-rays (3k+ px) make PIL resizing very slow. 512px is plenty for a heatmap.
        max_dim = 512
        w, h = image.size
        scale = min(max_dim / w, max_dim / h)
        if scale < 1.0:
            target_size = (int(w * scale), int(h * scale))
        else:
            target_size = (w, h)
            
        # 3. Create heatmap
        cam_img = Image.fromarray(np.uint8(255 * cam)).resize(target_size, Image.Resampling.BILINEAR)
        cam_resized = np.array(cam_img) / 255.0
        colormap = cm.get_cmap("jet")(cam_resized)[:, :, :3]
        heatmap = np.uint8(255 * colormap)
        
        # 4. Prepare base image (must match target_size)
        base_img = image.convert("RGB")
        if scale < 1.0:
            base_img = base_img.resize(target_size, Image.Resampling.LANCZOS)
        
        img_np = np.array(base_img)
        overlay = np.uint8(0.6 * img_np + 0.4 * heatmap)
        
        out_img = Image.fromarray(overlay)
        buf = io.BytesIO()
        out_img.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("utf-8")
    except Exception as e:
        logger.warning(f"CAM overlay generation failed: {e}")
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("utf-8")
"""
    lines[cam_start:cam_end] = [optimized_cam + "\n"]

# 3. Fix postprocess_probabilities (Clinical Logic)
# Remove the force-flagging of top result if it's below threshold
pp_start, pp_end = find_func_range(lines, "postprocess_probabilities")
if pp_start != -1:
    # We need to be careful with the range here as it's a longer function
    # Let's just find the specific lines to replace
    for i in range(pp_start, pp_end if pp_end != -1 else len(lines)):
        if "if not predicted_indices:" in lines[i]:
            # Replace the force-flagging block with a comment
            lines[i] = "        # Core Fix: Do not force-flag the top result if below clinical thresholds.\\n"
            lines[i+1] = "        # if not predicted_indices: predicted_indices = [int(np.argmax(probabilities))]\\n"
            break

# 4. Instrument run_pytorch_inference
inf_start, inf_end = find_func_range(lines, "run_pytorch_inference")
if inf_start != -1:
    # Add timing instrumentation
    for i in range(inf_start, inf_end if inf_end != -1 else len(lines)):
        if "logits = PYTORCH_MODEL(tensor)" in lines[i]:
             lines.insert(i+1, "        t_fwd = time.perf_counter()\\n")
        if "logits[0, class_idx].backward" in lines[i]:
             lines.insert(i+1, "        t_bwd = time.perf_counter()\\n")
        if "cam_b64 = generate_cam_overlay(image, cam_tensor)" in lines[i]:
             lines.insert(i+1, "        t_cam = time.perf_counter()\\n")
        if "return {" in lines[i]:
             # Add the debug timing to the return or log it
             lines.insert(i, "    logger.info(f\"TIMING: Fwd={round((t_fwd-start)*1000,1)}ms, Bwd={round((t_bwd-t_fwd)*1000,1)}ms, CAM={round((t_cam-t_bwd)*1000,1)}ms\")\\n")

with open(file_path, 'w') as f:
    f.writelines(lines)

print("Applied core latency and logic optimizations.")
