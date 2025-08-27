import os
import cv2
import torch
import numpy as np
import json, ast
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoProcessor, AutoModel, XCLIPModel
import os
import math
import uuid
import chromadb
from ultralytics import YOLO
import torch
import json
import base64
from openai import OpenAI, AsyncOpenAI
import asyncio
from concurrent.futures import ProcessPoolExecutor
from PIL import Image

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
CHUNK_SIZE = 10           # in seconds
FRAMES_PER_CHUNK = 8     # frames per chunk
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
COLLECTION_NAME = "chunk_captions"
client = chromadb.PersistentClient(path=OUTPUT_CHROMA_DIR)
collection = client.get_or_create_collection(name=COLLECTION_NAME, metadata={"hnsw:space": "cosine"})
VIDEO_COLLECTION_NAME = "video_captions"
video_collection = client.get_or_create_collection(name=VIDEO_COLLECTION_NAME, metadata={"hnsw:space": "cosine"})

peeks = collection.peek(50)

print(peeks['ids'])