import os
import sys
import warnings

# 1. SETUP: Suppress Warnings
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import cv2
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM, PreTrainedModel

# --- IMPORTS FOR OLLAMA ---
try:
    import ollama
except ImportError:
    print("❌ Error: 'ollama' library not found.")
    print("   Please run: pip install ollama")
    sys.exit(1)

# --- CONFIGURATION ---
VISION_MODEL_ID = "microsoft/Florence-2-base"
# Use the model you already have. 
# If your model is named "gemma:2b" or just "gemma", change this string accordingly.
LLM_MODEL_NAME = "qwen2.5:0.5b"  

def load_vision_model():
    print(f"⏳ Loading Vision Model ({VISION_MODEL_ID})...")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    
    # --- CRITICAL FIX FOR transformers ---
    if not hasattr(PreTrainedModel, "_supports_sdpa"):
        PreTrainedModel._supports_sdpa = False

    try:
        # 1. Load Vision Model (The Eye) only
        vision_model = AutoModelForCausalLM.from_pretrained(
            VISION_MODEL_ID, trust_remote_code=True, torch_dtype=dtype
        ).to(device).eval()
        processor = AutoProcessor.from_pretrained(VISION_MODEL_ID, trust_remote_code=True)

        print(f"✅ Vision Model loaded on {device.upper()}")
        return vision_model, processor, device
    except Exception as e:
        print(f"\n❌ Error loading vision model: {e}")
        sys.exit(1)

def analyze_frame(model, processor, device, frame_image):
    """Step 1: Ask Florence-2 to describe the image."""
    prompt = "<MORE_DETAILED_CAPTION>"
    
    inputs = processor(text=prompt, images=frame_image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    if device == "cuda":
        inputs["pixel_values"] = inputs["pixel_values"].to(torch.float16)

    generated_ids = model.generate(
        input_ids=inputs["input_ids"],
        pixel_values=inputs["pixel_values"],
        max_new_tokens=1024,
        do_sample=False,
        num_beams=3,
    )
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    
    parsed_answer = processor.post_process_generation(
        generated_text, 
        task=prompt, 
        image_size=(frame_image.width, frame_image.height)
    )
    return parsed_answer[prompt]

def check_intent_with_ollama(description):
    """Step 2: Ask local Ollama (Gemma 3) if the description sounds like theft."""
    
    prompt = f"""
You are a security AI. Analyze the scene description.
If the person is stealing, breaking a lock, or using tools like cutters/grinders aggressively, say THEFT.
If they are just walking, riding, or unlocking normally, say SAFE.
Output only one word.

Scene: {description}
"""
    
    try:
        response = ollama.chat(model=LLM_MODEL_NAME, messages=[
            {'role': 'user', 'content': prompt},
        ])
        
        # Get the clean text answer
        verdict = response['message']['content'].strip().upper()
        return verdict
    except Exception as e:
        return f"OLLAMA ERROR: {str(e)}"

def process_video(video_path):
    if not os.path.exists(video_path):
        print(f"❌ Error: Video file '{video_path}' not found.")
        return

    # Load only the vision model into Python memory
    v_model, processor, device = load_vision_model()
    
    # Verify Ollama is running
    try:
        print(f"⏳ Connecting to Ollama ({LLM_MODEL_NAME})...")
        ollama.list()
        print("✅ Ollama connected!")
    except:
        print("❌ Error: Could not connect to Ollama.")
        print("   Make sure the Ollama app is running in the background!")
        return

    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    if fps == 0: fps = 30 
    
    frame_count = 0
    seconds_processed = 0
    
    print(f"\n🎥 Watching {video_path}...")
    print("   Press 'q' in the video window to stop.\n")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # Analyze 1 frame every 1 second
        if frame_count % fps == 0:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_frame)
            
            try:
                # 1. Vision Phase (Florence-2)
                description = analyze_frame(v_model, processor, device, pil_image)
                
                # 2. Reasoning Phase (Ollama / Gemma 3)
                verdict = check_intent_with_ollama(description)
                
                # Logic check
                is_theft = "THEFT" in verdict
                status = "🚨 THEFT DETECTED" if is_theft else "✅ Normal Activity"
                color = (0, 0, 255) if is_theft else (0, 255, 0)
                
                print(f"[{seconds_processed}s] {status} ({verdict}): {description}")
                
                # Draw
                cv2.rectangle(frame, (0,0), (frame.shape[1], 80), (0,0,0), -1)
                cv2.putText(frame, f"AI Sees: {description[:60]}...", (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.putText(frame, f"AI Thinks: {verdict}", (10, 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
                cv2.putText(frame, status, (10, 75), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                
                cv2.imshow('AI Theft Detector', frame)
            except Exception as e:
                print(f"⚠️ Frame error: {e}")

            seconds_processed += 1

        frame_count += 1
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python theft_detector.py <video_file.mp4>")
    else:
        process_video(sys.argv[1])