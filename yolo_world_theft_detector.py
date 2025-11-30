import os
import sys
import time
import warnings
import cv2
import torch
import threading
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
    print("❌ Error: 'ollama' library not found.")
    sys.exit(1)

# --- CONFIGURATION ---
VISION_MODEL_ID = "microsoft/Florence-2-base"
# We use YOLO-World to detect custom objects without training
YOLO_MODEL_ID = "yolov8s-worldv2.pt" 
LLM_MODEL_NAME = "qwen2.5:0.5b"

# Define the custom classes we want YOLO to find
CUSTOM_CLASSES = ["person", "bicycle", "bolt cutter", "angle grinder", "hacksaw"]

# Thresholds
INTERSECTION_THRESHOLD = 0.05 

# Shared Variables
current_frame = None
latest_result = {
    "status": "Initializing...",
    "color": (255, 255, 0),
    "description": "",
    "boxes": [] 
}
frame_lock = threading.Lock()
stop_threads = False

def load_models():
    print(f"⏳ Loading AI Models on GPU...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    
    # 1. Load YOLO-World and teach it new words
    print(f"   - Loading YOLO-World (Scout)...")
    yolo_model = YOLO(YOLO_MODEL_ID)
    yolo_model.set_classes(CUSTOM_CLASSES) # <--- MAGIC HAPPENS HERE
    
    # 2. Load Florence-2
    print(f"   - Loading Florence-2 (Vision)...")
    if not hasattr(PreTrainedModel, "_supports_sdpa"):
        PreTrainedModel._supports_sdpa = False
    vision_model = AutoModelForCausalLM.from_pretrained(
        VISION_MODEL_ID, trust_remote_code=True, torch_dtype=dtype
    ).to(device).eval()
    processor = AutoProcessor.from_pretrained(VISION_MODEL_ID, trust_remote_code=True)
    
    return yolo_model, vision_model, processor, device

def analyze_intent_ollama(description):
    prompt = f"""Analyze: "{description}".
    Rules:
    - If cutting, breaking, sawing, bolt cutters -> THEFT
    - If picking up bike -> THEFT
    - If riding, walking, standing -> SAFE
    Output one word: THEFT or SAFE."""
    try:
        response = ollama.chat(model=LLM_MODEL_NAME, messages=[{'role':'user', 'content':prompt}])
        return response['message']['content'].strip().upper()
    except:
        return "SAFE"

def ai_worker():
    global latest_result
    yolo, v_model, processor, device = load_models()
    print("✅ AI Engine Ready!")
    
    while not stop_threads:
        with frame_lock:
            if current_frame is None:
                time.sleep(0.01)
                continue
            frame_copy = current_frame.copy()
        
        # 1. YOLO Detection
        # conf=0.15 allows detecting smaller tools that might be hard to see
        results = yolo(frame_copy, verbose=False, conf=0.15)[0]
        
        persons = []
        bikes = []
        tools = [] # New list for weapons/tools
        visual_boxes = []
        
        # Identify Objects
        for box in results.boxes:
            cls_id = int(box.cls[0])
            class_name = results.names[cls_id]
            coords = box.xyxy[0].cpu().numpy()
            x1, y1, x2, y2 = map(int, coords)
            
            if class_name == "person":
                persons.append(coords)
                visual_boxes.append((x1, y1, x2, y2, (255, 100, 0), "Person"))
            elif class_name == "bicycle":
                bikes.append(coords)
                visual_boxes.append((x1, y1, x2, y2, (0, 255, 0), "Bike"))
            elif class_name in ["bolt cutter", "angle grinder", "hacksaw"]:
                tools.append(class_name)
                visual_boxes.append((x1, y1, x2, y2, (0, 0, 255), f"⚠️ {class_name.upper()}"))

        # --- LOGIC BRANCHES ---
        status_text = "Scanning..."
        status_color = (255, 255, 0)
        desc_text = ""

        # PATH A: TOOL DETECTED (Instant Alarm)
        if len(tools) > 0:
            tool_name = tools[0]
            status_text = f"🚨 WEAPON DETECTED: {tool_name.upper()}"
            status_color = (0, 0, 255) # RED
            desc_text = f"System detected {tool_name}. Immediate Alert."
        
        # PATH B: INTERACTION DETECTED (Run Deep Analysis)
        else:
            interacting = False
            for p in persons:
                for b in bikes:
                    xA = max(p[0], b[0])
                    yA = max(p[1], b[1])
                    xB = min(p[2], b[2])
                    yB = min(p[3], b[3])
                    interArea = max(0, xB - xA) * max(0, yB - yA)
                    p_area = (p[2]-p[0]) * (p[3]-p[1])
                    
                    if p_area > 0 and (interArea / p_area) > INTERSECTION_THRESHOLD:
                        interacting = True
                        break # Stop checking other pairs
            
            if interacting:
                # Run Florence-2 Captioning
                rgb_frame = cv2.cvtColor(frame_copy, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(rgb_frame)
                
                prompt = "<CAPTION>"
                inputs = processor(text=prompt, images=pil_image, return_tensors="pt").to(device, torch.float16)
                
                gen_ids = v_model.generate(
                    input_ids=inputs["input_ids"], pixel_values=inputs["pixel_values"],
                    max_new_tokens=50, do_sample=False, num_beams=1
                )
                desc_text_raw = processor.batch_decode(gen_ids, skip_special_tokens=False)[0]
                parsed = processor.post_process_generation(desc_text_raw, task=prompt, image_size=pil_image.size)
                desc_text = parsed[prompt]
                
                # Run Brain
                intent = analyze_intent_ollama(desc_text)
                
                if "THEFT" in intent:
                    status_text = "🚨 THEFT DETECTED"
                    status_color = (0, 0, 255)
                else:
                    status_text = "✅ SAFE INTERACTION"
                    status_color = (0, 255, 0)
            else:
                status_text = "Safe (No Interaction)"
                status_color = (200, 200, 200)

        # Update Display Data
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

    t = threading.Thread(target=ai_worker)
    t.daemon = True
    t.start()
    
    cap = cv2.VideoCapture(video_path)
    print(f"\n🎥 Running YOLO-WORLD on {video_path}...")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: 
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue
        
        with frame_lock:
            current_frame = frame.copy()
            
        res = latest_result
        
        # Draw all boxes (People, Bikes, AND Tools)
        for (x1, y1, x2, y2, color, label) in res["boxes"]:
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
        # Draw UI
        cv2.rectangle(frame, (0,0), (800, 80), (0,0,0), -1)
        cv2.putText(frame, res["status"], (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, res["color"], 2)
        
        clean_desc = res["description"].replace("<pad>", "").replace("</s>", "")
        if len(clean_desc) > 50: clean_desc = clean_desc[:50] + "..."
        cv2.putText(frame, clean_desc, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        cv2.imshow('YOLO-World Theft Detector', frame)
        
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

    stop_threads = True
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python yolo_world_theft_detector.py <video_file.mp4>")
    else:
        process_video(sys.argv[1])