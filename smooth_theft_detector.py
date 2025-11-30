import os
import sys
import time
import warnings
import cv2
import torch
import threading
import queue
import numpy as np
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM, PreTrainedModel
from ultralytics import YOLO

# 1. SETUP
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
warnings.filterwarnings("ignore")

try:
    import ollama
except ImportError:
    print("❌ Error: 'ollama' library not found. Run: pip install ollama")
    sys.exit(1)

# --- CONFIGURATION ---
VISION_MODEL_ID = "microsoft/Florence-2-base"
YOLO_MODEL_ID = "yolo11n.pt"
LLM_MODEL_NAME = "qwen2.5:0.5b"
INTERSECTION_THRESHOLD = 0.05 

# Shared Variables (The "Bridge" between threads)
current_frame = None
latest_result = {
    "status": "Initializing...",
    "color": (255, 255, 0),
    "description": "",
    "boxes": [] # List of (x1, y1, x2, y2, color)
}
frame_lock = threading.Lock()
stop_threads = False

def load_models():
    print(f"⏳ Loading AI Models on GPU...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    
    # YOLO
    yolo_model = YOLO(YOLO_MODEL_ID)
    
    # Florence
    if not hasattr(PreTrainedModel, "_supports_sdpa"):
        PreTrainedModel._supports_sdpa = False
    vision_model = AutoModelForCausalLM.from_pretrained(
        VISION_MODEL_ID, trust_remote_code=True, torch_dtype=dtype
    ).to(device).eval()
    processor = AutoProcessor.from_pretrained(VISION_MODEL_ID, trust_remote_code=True)
    
    return yolo_model, vision_model, processor, device

def analyze_intent_ollama(description):
    """The slow thinking part"""
    prompt = f"""Analyze: "{description}".
    Rules:
    - If cutting, breaking, sawing, bolt cutters, pliers -> THEFT
    - If picking up bike from ground or carrying it -> THEFT
    - If "fixing", "repairing", "adjusting", or "tampering" with wheel/lock -> THEFT (Suspicious behavior)
    - If riding, walking, standing, parking -> SAFE
    Output one word: THEFT or SAFE."""
    try:
        response = ollama.chat(model=LLM_MODEL_NAME, messages=[{'role':'user', 'content':prompt}])
        return response['message']['content'].strip().upper()
    except:
        return "SAFE"

def ai_worker():
    """Background thread that runs the heavy AI"""
    global latest_result
    
    # Load models inside the thread
    yolo, v_model, processor, device = load_models()
    print("✅ AI Engine Started!")
    
    while not stop_threads:
        # 1. Get the latest frame (Skip old ones)
        with frame_lock:
            if current_frame is None:
                time.sleep(0.01)
                continue
            # Make a copy so we don't block the video player
            frame_copy = current_frame.copy()
        
        # 2. YOLO Detection (Fast)
        results = yolo(frame_copy, verbose=False, conf=0.3)[0]
        
        persons = []
        bikes = []
        visual_boxes = []
        
        for box in results.boxes:
            cls_id = int(box.cls[0])
            coords = box.xyxy[0].cpu().numpy()
            x1, y1, x2, y2 = map(int, coords)
            
            if cls_id == 0: # Person
                persons.append(coords)
                visual_boxes.append((x1, y1, x2, y2, (255, 100, 0))) # Blue
            elif cls_id == 1: # Bike
                bikes.append(coords)
                visual_boxes.append((x1, y1, x2, y2, (0, 255, 0))) # Green

        # 3. Intersection Logic
        interacting = False
        for p in persons:
            for b in bikes:
                # Simple intersection check
                xA = max(p[0], b[0])
                yA = max(p[1], b[1])
                xB = min(p[2], b[2])
                yB = min(p[3], b[3])
                interArea = max(0, xB - xA) * max(0, yB - yA)
                
                # Check if overlap is significant (>5% of person area)
                p_area = (p[2]-p[0]) * (p[3]-p[1])
                if p_area > 0 and (interArea / p_area) > INTERSECTION_THRESHOLD:
                    interacting = True
                    # Mark interaction box in Red
                    visual_boxes.append((int(p[0]), int(p[1]), int(p[2]), int(p[3]), (0, 0, 255)))

        # 4. Deep Analysis (Only if interacting)
        status_text = "Scanning..."
        status_color = (255, 255, 0)
        desc_text = ""

        if interacting:
            # Prepare image for Florence
            rgb_frame = cv2.cvtColor(frame_copy, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_frame)
            
            # Vision Caption
            prompt = "<CAPTION>"
            inputs = processor(text=prompt, images=pil_image, return_tensors="pt").to(device, torch.float16)
            
            gen_ids = v_model.generate(
                input_ids=inputs["input_ids"], pixel_values=inputs["pixel_values"],
                max_new_tokens=50, do_sample=False, num_beams=1
            )
            desc_text = processor.batch_decode(gen_ids, skip_special_tokens=False)[0]
            parsed = processor.post_process_generation(desc_text, task=prompt, image_size=pil_image.size)
            final_desc = parsed[prompt]
            
            # Brain Logic
            intent = analyze_intent_ollama(final_desc)
            
            if "THEFT" in intent:
                status_text = "🚨 THEFT DETECTED"
                status_color = (0, 0, 255)
            else:
                status_text = "✅ SAFE INTERACTION"
                status_color = (0, 255, 0)
        
        # 5. Update Global Results (Instant)
        latest_result = {
            "status": status_text,
            "color": status_color,
            "description": desc_text,
            "boxes": visual_boxes
        }

def process_video(video_path):
    global current_frame, stop_threads
    
    if not os.path.exists(video_path):
        print("❌ File not found.")
        return

    # Start AI Thread
    t = threading.Thread(target=ai_worker)
    t.daemon = True # Kill thread when main program ends
    t.start()
    
    cap = cv2.VideoCapture(video_path)
    
    print(f"\n🎥 Playing {video_path}...")
    print("   The AI is initializing in the background (Wait 10-20s)...")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: 
            # Loop video for testing
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue
        
        # Update shared frame for the AI thread
        with frame_lock:
            current_frame = frame.copy()
            
        # --- DRAWING (Fast, runs at 30 FPS) ---
        # We draw whatever the LATEST result from the AI thread is.
        # It might be 5 frames old, but it keeps the video smooth.
        res = latest_result
        
        # Draw boxes
        for (x1, y1, x2, y2, color) in res["boxes"]:
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
        # Draw UI
        cv2.rectangle(frame, (0,0), (800, 80), (0,0,0), -1)
        cv2.putText(frame, res["status"], (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, res["color"], 2)
        
        # Show cleaned up description
        clean_desc = res["description"].replace("<pad>", "").replace("</s>", "")
        if len(clean_desc) > 50: clean_desc = clean_desc[:50] + "..."
        cv2.putText(frame, clean_desc, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        cv2.imshow('Smooth Theft Detector', frame)
        
        # 30ms delay = ~30 FPS playback
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

    stop_threads = True
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python smooth_theft_detector.py <video_file.mp4>")
    else:
        process_video(sys.argv[1])