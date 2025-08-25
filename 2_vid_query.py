import os
import cv2
import torch
import numpy as np
import json
from transformers import AutoProcessor, AutoModel, XCLIPModel, AutoTokenizer
import os
import math
import uuid
import chromadb
from FlagEmbedding import FlagReranker

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
RERANKER_MODEL_NAME = "BAAI/bge-reranker-base"

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
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=False)
model = AutoModel.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    trust_remote_code=True).to(device)
model.eval()

# --- Load reranker (cross-encoder) ---
reranker = FlagReranker(RERANKER_MODEL_NAME)

# -------------------------
# --- Connect to Chroma ---
# -------------------------
client = chromadb.PersistentClient(path=CHROMA_PATH)
collection = client.get_or_create_collection(name=COLLECTION_NAME, metadata={"hnsw:space": "cosine"})

# --- Text embedding function ---
def get_text_embedding(text, device="cpu"):
    inputs = tokenizer(text, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.language_model(**inputs, output_hidden_states=True)
        last_hidden_state = outputs.hidden_states[-1]
        text_emb_1536d = last_hidden_state.mean(dim=1)
        text_emb_1024d = model.text_projection(text_emb_1536d)
        emb = text_emb_1024d / text_emb_1024d.norm(dim=-1, keepdim=True)
        return emb[0].to(torch.float32).cpu().numpy()

# --- Search function ---
def search_videos(query, top_k=10, rerank_top_n=5):
    query_emb = get_text_embedding(query, device=device)
    results = collection.query(
        query_embeddings=[query_emb],
        n_results=top_k,
        include=["embeddings", "documents", "metadatas", "distances"]
    )

    docs = results["documents"][0]
    ids = results["ids"][0]
    metas = results["metadatas"][0]

    if not docs:
        return []

    return results

query = "a man in white shirt and backpack standing at bus stop"
results = search_videos(query, top_k=10, rerank_top_n=5)

print(f"\n🔎 Query: {query}\n")
for i, (doc, meta, dist, vid) in enumerate(
    zip(results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
        results["ids"][0])
):
    print(f"Result {i+1}:")
    print(f"  ID        : {vid}")
    print(f"  Document  : {doc}")
    print(f"  Metadata  : {meta}")
    print(f"  Distance  : {1-dist:.4f}")
    print("-" * 50)