Real-Time AI Bicycle Theft Detector

This project implements a multi-stage AI pipeline to detect bicycle theft in real-time on consumer hardware (e.g., laptops with 4GB VRAM). It goes beyond simple object detection by analyzing human behavior and intent.

🚀 Key Features

Real-Time Processing: Uses threading to ensure smooth video playback (30 FPS) while AI runs in the background.

Behavior Analysis: Distinguishes between "Riding" (Safe) and "Stealing/Cutting" (Theft).

Edge Computing: Optimized to run locally on NVIDIA GTX 1650/3050 GPUs without cloud costs.

🛠️ The Two Versions

1. Standard Detector (smooth_theft_detector.py)

Best for: General theft detection based on behavior.

Logic: 1. Detects Person + Bike using YOLOv11.
2. Checks if they overlap (Intersection).
3. Uses Florence-2 to describe the action ("Man crouching...").
4. Uses Ollama (Qwen) to decide if the action is suspicious.

2. YOLO-World Detector (yolo_world_theft_detector.py)

Best for: Detecting tools and weapons instantly.

Logic:

Uses YOLO-World (Open Vocabulary) to find: Person, Bike, Bolt Cutter, Angle Grinder, Hacksaw.

Instant Alarm: If a tool is seen, it alerts immediately without waiting for behavior analysis.

Behavior Analysis: If no tool is seen but a person touches a bike, it falls back to the Florence-2/Ollama logic.

⚙️ Setup & Installation

Prerequisites

Hardware: NVIDIA GPU (4GB VRAM recommended), 16GB System RAM.

Software: Python 3.10+, Ollama installed.

1. Install Dependencies

Run this command to install the required Python libraries:

pip install ultralytics transformers==4.45.2 einops timm accelerate ollama opencv-python pillow numpy


2. Install GPU-Accelerated PyTorch

Crucial Step: Standard pip often installs the CPU version. Run this to force GPU support:

pip install torch torchvision torchaudio --index-url [https://download.pytorch.org/whl/cu118](https://download.pytorch.org/whl/cu118)


3. Setup Ollama

Download the lightweight "Brain" model:

ollama pull qwen2.5:0.5b


▶️ How to Run

Run the Standard Detector

python smooth_theft_detector.py your_video.mp4


Run the YOLO-World Detector (Weapon Detection)

python yolo_world_theft_detector.py your_video.mp4


(Replace your_video.mp4 with the path to your video file)

🏗️ Architecture

Component

Model

Role

The Scout

YOLOv11 / YOLO-World

Fast object detection (100+ FPS). Finds people, bikes, and tools.

The Logic

Python (IoU)

Geometry check. Only triggers AI if Person & Bike bounding boxes intersect.

The Eye

Florence-2-base

Vision-Language Model. "Looks" at the specific interaction and generates a text description.

The Brain

Qwen 2.5 (0.5B)

Large Language Model. Reads the text description and judges "Safe" vs "Theft" based on security rules.

❓ Troubleshooting

"Ollama Error: Model requires more system memory"

Your system RAM is full. Ensure you are using qwen2.5:0.5b (0.4GB) and not gemma3 (4GB).

"System is using CPU"

Run python check_gpu.py. If it says False, reinstall PyTorch using the "Force-Reinstall" command in Step 2.
