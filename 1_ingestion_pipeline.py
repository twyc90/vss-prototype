import os
import cv2
import torch
import numpy as np
import json
from sentence_transformers import SentenceTransformer
from transformers import AutoProcessor, AutoModel, XCLIPModel
import os
import math
import uuid
import chromadb
from ultralytics import YOLO
import torch
import json
import base64
from openai import OpenAI

# --------------
# --- Config ---
# --------------
DATA_DIR = "./data/videos"
OUTPUT_DIR = "./out"
OUTPUT_CHUNK_DIR = "./out/video_chunks"
ANNOTATED_CHUNK_DIR = "./out/video_chunks_cv"
OUTPUT_METADATA_DIR = "./out/metadata"
OUTPUT_CHROMA_DIR = "./out/chroma_db"
SAVE_CHUNKS = True
CHUNK_SIZE = 8            # in seconds
FRAMES_PER_CHUNK = 10     # frames per chunk
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(OUTPUT_CHUNK_DIR, exist_ok=True)
os.makedirs(ANNOTATED_CHUNK_DIR, exist_ok=True)
os.makedirs(OUTPUT_METADATA_DIR, exist_ok=True)
os.makedirs(OUTPUT_CHROMA_DIR, exist_ok=True)

# -------------------------------------
# --- Device selection (MPS on Mac) ---
# -------------------------------------
if torch.backends.mps.is_available():
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

# -------------------------
# --- Connect to Chroma ---
# -------------------------
COLLECTION_NAME = "video_embeddings"
client = chromadb.PersistentClient(path=OUTPUT_CHROMA_DIR)
collection = client.get_or_create_collection(name=COLLECTION_NAME, metadata={"hnsw:space": "cosine"})

# ------------------------
# --- Emb Model Config ---
# ------------------------
EMBEDDING_MODEL_NAME = 'BAAI/bge-m3'
embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)

# ------------------------------
# --- YOLO Obj Detect Config ---
# ------------------------------
YOLO_MODEL = "yolo11n.pt"
model = YOLO(YOLO_MODEL)
CONFIDENCE_THRESHOLD = 0.6

# ------------------
# --- LLM Config ---
# ------------------
VLM_API_BASE = "http://localhost:1234/v1"
VLM_API_KEY = "NA"
# VLM_MODEL_NAME = "internvl3-14b-instruct"
# VLM_MODEL_NAME = "google/gemma-3-12b"
VLM_MODEL_NAME = "google/gemma-3-27b"
# VLM_MODEL_NAME = "qwen/qwen2.5-vl-7b"
client = OpenAI(base_url=VLM_API_BASE, api_key=VLM_API_KEY)

# ------------------------
# --- Helper Functions ---
# ------------------------
def chunk_video(video_path):
    print(f"Chunking video: {video_path}")
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    chunk_frames = int(fps * CHUNK_SIZE)
    frame_count = 0
    chunk_count = 0
    video_filename = os.path.basename(video_path)
    video_name, video_ext = os.path.splitext(video_filename)
    while True:
        frames = []
        for _ in range(chunk_frames):
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        if not frames:
            break
        chunk_filename = f"{video_name}_chunk_{chunk_count}{video_ext}"
        chunk_path = os.path.join(OUTPUT_CHUNK_DIR, chunk_filename)
        fourcc = cv2.VideoWriter_fourcc(*'avc1')
        out = cv2.VideoWriter(chunk_path, fourcc, fps, (frames[0].shape[1], frames[0].shape[0]))
        for frame in frames:
            out.write(frame)
        out.release()
        chunk_count += 1
    cap.release()

def detect_motion(chunk_path):
    # print(f"Detecting motion: {video_path}")
    cap = cv2.VideoCapture(chunk_path)
    backSub = cv2.createBackgroundSubtractorMOG2()
    motion_detected = False
    frame_count = 0
    while frame_count < 150:
        ret, frame = cap.read()
        if not ret:
            break
        fgMask = backSub.apply(frame)
        if np.sum(fgMask > 200) > (frame.shape[0] * frame.shape[1] * 0.001):
            motion_detected = True
            break
        frame_count += 1
    cap.release()
    return motion_detected

def detect_and_track_objects(chunk_path, model):
    print(f"Detect motion & track object: {video_path}")
    cap = cv2.VideoCapture(chunk_path)

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    output_video_path = os.path.join(ANNOTATED_CHUNK_DIR, os.path.basename(chunk_path))
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    chunk_metadata = {
        "chunk_path": chunk_path,
        "motion_detected": detect_motion(chunk_path),
        "objects": []
    }
    if not detect_motion:
        return 

    tracked_objects = {}
    next_object_id = 0
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        latest_boxes_to_draw = []
        NTH_FRAME = 3
        if frame_idx % NTH_FRAME == 0:
            results = model(frame, verbose=False)
            for result in results:
                for box in result.boxes:
                    if box.conf[0] > CONFIDENCE_THRESHOLD:
                        coords_float = box.xyxy[0].cpu().numpy()
                        coords = coords_float.astype(int)
                        class_id = int(box.cls[0].cpu().numpy())
                        class_name = model.names[class_id]
                        center_x = (coords_float[0] + coords_float[2]) / 2
                        center_y = (coords_float[1] + coords_float[3]) / 2
                        
                        confidence = box.conf[0].cpu().numpy().item()
                        found_match = False
                        display_id = -1
                        for obj_id, obj_data in tracked_objects.items():
                            dist = np.sqrt((center_x - obj_data['last_pos'][0])**2 + (center_y - obj_data['last_pos'][1])**2)
                            if obj_data['label'] == class_name and dist < 75:
                                obj_data['last_pos'] = (center_x, center_y)
                                if frame_idx not in obj_data['frames_present']:
                                    obj_data['frames_present'].append(frame_idx)
                                found_match = True
                                display_id = obj_id
                                break
                        
                        if not found_match:
                            display_id = next_object_id
                            tracked_objects[display_id] = {
                                'id': display_id,
                                'label': class_name,
                                'initial_pos': (center_x, center_y),
                                'last_pos': (center_x, center_y),
                                'frames_present': [frame_idx],
                            }
                            tracked_objects[display_id]['last_confidence'] = confidence
                            next_object_id += 1

                        label = f"ID {display_id}: {class_name} {box.conf[0]:.2f}"
                        latest_boxes_to_draw.append((coords, label))
        for coords, label in latest_boxes_to_draw:
             cv2.rectangle(frame, (coords[0], coords[1]), (coords[2], coords[3]), (0, 255, 0), 2)
             cv2.putText(frame, label, (coords[0], coords[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        out.write(frame)
        frame_idx += 1
    chunk_metadata["objects"] = list(tracked_objects.values())
    metadata_filename = os.path.basename(chunk_path).replace(os.path.splitext(chunk_path)[1], '.json')
    metadata_path = os.path.join(OUTPUT_METADATA_DIR, metadata_filename)
    with open(metadata_path, 'w') as f:
        def convert(o):
            if isinstance(o, np.generic): return o.item()  
            raise TypeError
        json.dump(chunk_metadata, f, indent=4, default=convert)
    
    cap.release()
    out.release()

def vlm_caption(chunk_path, client):
    print(f"Generating caption for: {chunk_path}")
    cap = cv2.VideoCapture(chunk_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    base64_frames = []
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # Sample 1 frame per second
        if frame_count % int(fps) == 0:
            _, buffer = cv2.imencode(".jpg", frame)
            base64_frames.append(base64.b64encode(buffer).decode("utf-8"))
        frame_count += 1
    cap.release()
    if not base64_frames:
        return "No frames extracted for captioning."
    try:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": (
        "These are sequential frames from a video clip."
        "Objects detected are bounded by a green color box."
        "You are now expert investigation officer that looks for distinguished and/or suspicious feature from video clips for national security concerns."
        "1. Identify all the individuals and determine their features such as clothing, appearance, etc and deduce what they are doing."
        "2. Identify all objects like car, personal mobility devices (PMD), bicycle, scooter, etc and record their color and features."
        "2. Identify suspicious objects like unclaimed bag in the middle of a bus stop that looked suspicious, etc and provide the description"
        "3. Record the likely location of the video clip, such as bus stop, shopping mall, restaurant, MRT, MBS, etc"
        "4. Generate a concise one sentence caption for the video clip."
        "Return your result in a valid JSON output as per below:"
        '''
        "Features": [
        {
        "tracker": "Young Man",
        "description": "A young man wearing a backpack, jeans, and a light-colored shirt. He appears to be looking at his phone. There is a red color handbag on the floor near to him.",
        "location": "Standing near the edge of a bus stop shelter.",
        },
        {
        "tracker": "Bus",
        ""description": "A green color SG bus with number 298",
        "location": "On the road",
        },
        {
        "tracker": "Car",
        ""description": "A bright yellow color Honda civic",
        "location": "On the road",
        },
        {
        "tracker": "Elderly Woman",
        ""description": "An elderly woman wearing a dark jacket and seated on the bus stop bench.",
        "location": "Seated on the bus stop bench."
        }],
        "Video Summary": 
        "this video clip is in a restaurant setting, there are two individual in the video."
                }'''
                    )},
                    *[{"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_str}"}} for img_str in base64_frames]
                ]
            },
        ]
        response = client.chat.completions.create(
            model=VLM_MODEL_NAME,
            messages=messages,
            max_tokens=4096,
        )
        caption = response.choices[0].message.content
        # print(f"-"*25)
        # print(f"Generated Caption: {caption}")
    except Exception as e:
        print(f"Error during OpenAI API call: {e}")
        caption="Caption generation failed."

    metadata_filename = chunk_filename.replace(os.path.splitext(chunk_filename)[1], '.json')
    metadata_path = os.path.join(OUTPUT_METADATA_DIR, metadata_filename)
    with open(metadata_path, 'r+') as f:
        data = json.load(f)
        data['vlm_caption'] = caption
        f.seek(0)
        json.dump(data, f, indent=4)
        f.truncate()

    embedding = embedding_model.encode(EMBEDDING_MODEL_NAME, normalize_embeddings=True)
    embedding = embedding.tolist()
    if embedding:
        try:
            collection.upsert(
                embeddings=[embedding],
                metadatas=[{"chunk_path": chunk_path, "caption": caption}],
                ids=[chunk_filename] # Use filename as a unique ID
            )
            print(f"Added embedding for {chunk_filename} to ChromaDB.")
        except Exception as e:
            print(f"Error adding to ChromaDB: {e}")
    return caption

## Chunk Video
for filename in os.listdir(DATA_DIR):
    if filename.lower().endswith((".mp4", ".avi", ".mov")):
        video_path = os.path.join(DATA_DIR, filename)
        chunk_video(video_path)

## CV Pipeline
for chunk_filename in os.listdir(OUTPUT_CHUNK_DIR):
    if chunk_filename.lower().endswith((".mp4", ".avi", ".mov")):
        chunk_path = os.path.join(OUTPUT_CHUNK_DIR, chunk_filename)
        detect_and_track_objects(chunk_path, model)
    os.remove(chunk_path)

## VLM Pipeline
for chunk_filename in os.listdir(ANNOTATED_CHUNK_DIR):
    if chunk_filename.lower().endswith((".mp4", ".avi", ".mov")):
        chunk_path = os.path.join(ANNOTATED_CHUNK_DIR, chunk_filename)
        vlm_caption(chunk_path, client)
        
