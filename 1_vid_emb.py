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

# --- Config ---
VIDEO_DIR = "./data/videos"
NUM_FRAMES = 16
CHUNK_DURATION = 16
OUTPUT_PATH = "./out"
OUTPUT_EMB = "./out/video_embeddings.npy"
OUTPUT_INDEX = "./out/video_index.json"
os.makedirs(OUTPUT_PATH, exist_ok=True)
CHROMA_PATH = "./out/chroma_db"
COLLECTION_NAME = "video_embeddings"

# --- Device selection (MPS on Mac) ---
if torch.backends.mps.is_available():
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

print(f"Using device: {device}")

# --- Load model ---
MODEL_ID = "microsoft/xclip-base-patch16-16-frames"
# MODEL_ID = "OpenGVLab/InternVideo2-CLIP-6B-224p-f8" #requires flash attention, CUDA
processor = AutoProcessor.from_pretrained(MODEL_ID)
model = AutoModel.from_pretrained(MODEL_ID).to(device)
model.eval()

# --- Connect to Chroma ---
client = chromadb.PersistentClient(path=CHROMA_PATH)
collection = client.get_or_create_collection(name=COLLECTION_NAME, metadata={"hnsw:space": "cosine"})

# --- Sample frames at 1 FPS for a chunk ---
def sample_chunk_frames_1fps(cap, start_sec, end_sec, num_frames=16):
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = []

    for sec in range(start_sec, end_sec):
        frame_id = int(sec * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        ret, frame = cap.read()
        if not ret:
            continue
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame_rgb)

    if len(frames) == 0:
        return []

    # If too many frames → downsample
    if len(frames) > num_frames:
        indices = np.linspace(0, len(frames) - 1, num_frames, dtype=int)
        frames = [frames[i] for i in indices]

    # If too few frames → pad by repeating
    elif len(frames) < num_frames:
        while len(frames) < num_frames:
            frames.append(frames[-1])  # repeat last frame

    return frames

# --- Embedding function ---
def get_chunk_embedding(frames):
    if len(frames) == 0:
        return None
    inputs = processor(videos=frames, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.get_video_features(**inputs)
    emb = outputs / outputs.norm(dim=-1, keepdim=True)  # normalize
    return emb[0].cpu().numpy()

# --- Process one video into chunks ---
def process_video(video_path, chunk_duration=60, num_frames=16):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0 or total_frames == 0:
        cap.release()
        return []

    total_secs = int(total_frames / fps)
    num_chunks = max(1, math.ceil(total_secs / chunk_duration))

    embeddings = []
    for c in range(num_chunks):
        start_sec = c * chunk_duration
        end_sec = min((c + 1) * chunk_duration, total_secs)
        frames = sample_chunk_frames_1fps(cap, start_sec, end_sec, num_frames)
        if len(frames) == 0:
            continue
        emb = get_chunk_embedding(frames)
        if emb is not None:
            embeddings.append((c, start_sec, end_sec, emb))

    cap.release()
    return embeddings

# --- Main loop over folder ---
all_embeddings = []
video_index = {}

for i, fname in enumerate(os.listdir(VIDEO_DIR)):
    if not fname.lower().endswith((".mp4", ".mov", ".avi", ".mkv")):
        continue

    path = os.path.join(VIDEO_DIR, fname)
    print(f"Processing {fname} ...")

    chunk_embs = process_video(path, CHUNK_DURATION, NUM_FRAMES)
    for chunk_id, start, end, emb in chunk_embs:
        doc_id = f"{fname}_{chunk_id}"
        collection.upsert(
            ids=[doc_id],
            embeddings=[emb],
            metadatas=[{
                "video_name": fname,
                "chunk_id": chunk_id,
                "start_time_sec": chunk_id * CHUNK_DURATION,
                "end_time_sec": (chunk_id + 1) * CHUNK_DURATION,
            }],
            documents=[f"Video: {fname}, Chunk: {chunk_id}"]
        )
print("✅ Done. Saved embeddings to index")