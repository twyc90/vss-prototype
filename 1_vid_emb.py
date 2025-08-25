import os
import cv2
import torch
import numpy as np
import json
from transformers import AutoProcessor, AutoModel, XCLIPModel
import os
import math
import uuid
import chromadb

# --------------
# --- Config ---
# --------------
VIDEO_PATH = "./data/videos"
NUM_FRAMES = 16
CHUNK_DURATION = 16
OUTPUT_PATH = "./out"
OUTPUT_EMB = "./out/video_embeddings.npy"
OUTPUT_INDEX = "./out/video_index.json"
VIDEO_CHUNKS_PATH = "./out/video_chunks"
os.makedirs(OUTPUT_PATH, exist_ok=True)
os.makedirs(VIDEO_CHUNKS_PATH, exist_ok=True)
CHROMA_PATH = "./out/chroma_db"
COLLECTION_NAME = "video_embeddings"
SAVE_CHUNKS = True  # Toggle to enable/disable chunk saving

# -------------------------------------
# --- Device selection (MPS on Mac) ---
# -------------------------------------
if torch.backends.mps.is_available():
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
print(f"Using device: {device}")

# ------------------
# --- Load model ---
# ------------------
from transformers import AutoImageProcessor
MODEL_ID = "OpenGVLab/InternVL3-2B"
processor = AutoImageProcessor.from_pretrained(MODEL_ID)
model = AutoModel.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    trust_remote_code=True).to(device)
model.eval()

# -------------------------
# --- Connect to Chroma ---
# -------------------------
client = chromadb.PersistentClient(path=CHROMA_PATH)
collection = client.get_or_create_collection(name=COLLECTION_NAME, metadata={"hnsw:space": "cosine"})

# ------------------------
# --- Helper Functions ---
# ------------------------
# --- Sample frames at 1 FPS for a chunk ---
def sample_chunk_frames_1fps(cap, start_sec, end_sec, num_frames=16, save=False, video_name=None, chunk_id=None):
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames = []
    
    # Prepare VideoWriter if saving is enabled
    out = None
    chunk_path = None
    if save and video_name is not None and chunk_id is not None:
        base_name = os.path.splitext(video_name)[0]
        chunk_filename = f"{base_name}_chunk_{chunk_id:03d}_{start_sec:04d}s-{end_sec:04d}s_1fps.mp4"
        chunk_path = os.path.join(VIDEO_CHUNKS_PATH, chunk_filename)
        fourcc = cv2.VideoWriter_fourcc(*'avc1')
        out = cv2.VideoWriter(chunk_path, fourcc, 1.0, (width, height))  # force 1 FPS

    # Sample 1 frame per second
    for sec in range(start_sec, end_sec):
        frame_id = int(sec * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        ret, frame = cap.read()
        if not ret:
            continue
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame_rgb)
        if out is not None:
            out.write(frame)  # save sampled frame into 1 FPS video
    # Release VideoWriter
    if out is not None:
        out.release()

    # If too many frames → downsample
    if len(frames) > num_frames:
        indices = np.linspace(0, len(frames) - 1, num_frames, dtype=int)
        frames = [frames[i] for i in indices]

    # If too few frames → pad by repeating
    elif len(frames) < num_frames:
        while len(frames) < num_frames:
            frames.append(frames[-1])  # repeat last frame

    return frames, chunk_path

# --- Embedding function ---
def get_chunk_embedding(frames):
    if len(frames) == 0:
        return None
    inputs = processor(images=frames, return_tensors="pt", padding=True).to(device)
    inputs['pixel_values'] = inputs['pixel_values'].to(torch.bfloat16)
    with torch.no_grad():
        vision_outputs = model.vision_model(**inputs)
        frame_embeddings = vision_outputs.pooler_output
        chunk_embedding = torch.mean(frame_embeddings, dim=0)
    normalized_emb = chunk_embedding / chunk_embedding.norm(dim=-1, keepdim=True)
    return normalized_emb.to(torch.float32).cpu().numpy()

# --- Process one video into chunks ---
def process_video(video_path, chunk_duration=16, num_frames=16):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0 or total_frames == 0:
        cap.release()
        return []

    total_secs = int(total_frames / fps)
    num_chunks = max(1, math.ceil(total_secs / chunk_duration))
    
    # Get video filename for chunk naming
    video_name = os.path.basename(video_path)

    embeddings = []
    for c in range(num_chunks):
        start_sec = c * chunk_duration
        end_sec = min((c + 1) * chunk_duration, total_secs)
        frames, chunk_path = sample_chunk_frames_1fps(cap, start_sec, end_sec, num_frames, save=SAVE_CHUNKS, video_name=video_name, chunk_id=c)
        if len(frames) == 0:
            continue
        emb = get_chunk_embedding(frames)
        if emb is not None:
            embeddings.append((c, start_sec, end_sec, emb, chunk_path))

    cap.release()
    return embeddings

# -----------------------------
# --- Main loop over folder ---
# -----------------------------
all_embeddings = []
video_index = {}

for i, fname in enumerate(os.listdir(VIDEO_PATH)):
    if not fname.lower().endswith((".mp4", ".mov", ".avi", ".mkv")):
        continue

    path = os.path.join(VIDEO_PATH, fname)
    print(f"Processing {fname} ...")

    chunk_embs = process_video(path, CHUNK_DURATION, NUM_FRAMES)
    for chunk_data in chunk_embs:
        chunk_id, start, end, emb, chunk_path = chunk_data

        doc_id = f"{fname}_{chunk_id}"
        metadata = {
            "video_name": fname,
            "chunk_id": chunk_id,
            "start_time_sec": chunk_id * CHUNK_DURATION,
            "end_time_sec": (chunk_id + 1) * CHUNK_DURATION,
        }
        
        # Add chunk path to metadata if available
        if chunk_path:
            metadata["chunk_path"] = chunk_path
            print(f"  Saved chunk {chunk_id} to: {chunk_path}")
            
        collection.upsert(
            ids=[doc_id],
            embeddings=[emb],
            metadatas=[metadata],
            documents=[f"Video: {fname}, Chunk: {chunk_id}"]
        )
        
print("✅ Done. Saved embeddings to index")
if SAVE_CHUNKS:
    print(f"✅ Video chunks saved to: {VIDEO_CHUNKS_PATH}")
