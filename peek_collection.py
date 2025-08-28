import os
import cv2
import torch
import numpy as np
import json, ast
from FlagEmbedding import FlagReranker
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
OUTPUT_CHROMA_DIR = "./out/chroma_db"
COLLECTION_NAME = "chunk_captions"
# EMBEDDING_MODEL_NAME = 'BAAI/bge-m3'
EMBEDDING_MODEL_NAME = "BAAI/BGE-VL-base"
RERANKER_MODEL_NAME = 'BAAI/bge-reranker-large'
ANNOTATED_CHUNK_DIR = "./out/video_chunks_cv"
OUTPUT_METADATA_DIR = "./out/metadata"
VIDEO_COLLECTION_NAME = "video_captions"
MIN_SCORE = 0.6
COLOR_SCORE = 0.5
RERANKER_TOP_K = 5

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


def encode_text(texts, processor, model, device="cpu", normalize=True, max_length=77):
    embeddings = []
    try:
        for text in texts:
            tokens = processor(text=text, return_tensors="pt", add_special_tokens=True)
            input_ids = tokens["input_ids"][0]
            if len(input_ids) <= max_length:
                print('normal encode')
                inputs = processor(text=text, return_tensors="pt", padding=True, truncation=True, max_length=max_length).to(device)
                with torch.no_grad():
                    outputs = model.get_text_features(**inputs)
                emb = outputs.cpu().numpy()
            else:
                print('chunk and mean encode')
                chunks = [input_ids[i:i+max_length] for i in range(0, len(input_ids), max_length)]
                chunk_embs = []
                for chunk in chunks:
                    inputs = {"input_ids": chunk.unsqueeze(0).to(device)}
                    with torch.no_grad():
                        outputs = model.get_text_features(**inputs)
                    chunk_embs.append(outputs.cpu().numpy())
                emb = np.mean(chunk_embs, axis=0)
            if normalize:
                emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)
    except Exception as e:
        print(f'Encoding error. {e}')
    return emb.tolist()


def encode_image(images, processor, model, device="cpu", normalize=True):
    try:
        inputs = processor(images=images, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model.get_image_features(**inputs)
        embeddings = outputs.cpu().numpy()
        if normalize:
            embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    except Exception as e:
        print(f'Encoding error. {e}')
    return embeddings.tolist()


def encode_video(pil_frames, processor, model, device="cpu", normalize=True):
    try:
        image_embeddings = encode_image(pil_frames, processor, embedding_model, device=device)
        if len(image_embeddings)>1:
            avg_image_embedding = np.mean(image_embeddings, axis=0)
            avg_image_embedding = avg_image_embedding / np.linalg.norm(avg_image_embedding)
            avg_image_embedding = avg_image_embedding.tolist()
        else:
            avg_image_embedding = image_embeddings[0]
    except Exception as e:
        print(f'Encoding error. {e}')
    return avg_image_embedding



def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    if norm_vec1 == 0 or norm_vec2 == 0:
        return 0
    return dot_product / (norm_vec1 * norm_vec2)


def load_models():
    # embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    processor = AutoProcessor.from_pretrained(EMBEDDING_MODEL_NAME)
    embedding_model = AutoModel.from_pretrained(EMBEDDING_MODEL_NAME).to(device)
    embedding_model.eval()
    reranker_model = FlagReranker(RERANKER_MODEL_NAME, use_fp16=True)
    print("Models loaded successfully.")
    return embedding_model, reranker_model, processor

embedding_model, reranker_model, processor = load_models()
# Text
query_text = 'Group of male and female subjects walking together. One of them wearing a T-shirt with strawberry motifs.'
query_embedding = encode_text([query_text], processor, embedding_model, device=device)[0]

# Image
# img = Image.open('/Users/yeecherngoh/Downloads/htx-vss-proj/data/videos_bckup/Screenshot 2025-08-27 at 10.33.21 PM.png')
# emb_image = encode_image([img], processor, embedding_model, device=device)[0]
# query_embedding = emb_image

# Video
# cap = cv2.VideoCapture('/Users/yeecherngoh/Downloads/htx-vss-proj/data/videos/CCTV4|VID_GEN1.mp4')
# fps = cap.get(cv2.CAP_PROP_FPS)
# all_frames = []
# while True:
#     ret, frame = cap.read()
#     if not ret: break
#     all_frames.append(frame)
# cap.release()
# num_frames = len(all_frames)
# indices = np.linspace(0, num_frames - 1, num=min(num_frames, 8), dtype=int)
# sampled_frames = [all_frames[i] for i in indices]
# pil_frames = [Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)) for frame in sampled_frames]
# query_embedding = encode_video(pil_frames, processor, embedding_model, device=device)

# Text
search_results = collection.query(
    query_embeddings=[query_embedding],
    n_results=5,
    include=['metadatas','documents','embeddings','distances'],
)
rerank_pairs = []
for metadata in search_results['metadatas'][0]:
    rerank_pairs.append([query_text, metadata.get('caption', '')])
rerank_scores = reranker_model.compute_score(rerank_pairs)
rerank_scores = torch.sigmoid(torch.tensor(rerank_scores)).tolist()
reranked_chunks = []
for i, (meta, emb) in enumerate(zip(search_results['metadatas'][0], search_results['embeddings'][0])):
    reranked_chunks.append({
        'metadata': meta,
        'rerank_score': rerank_scores[i],
        'embeddings': emb
    })
reranked_chunks.sort(key=lambda x: x['rerank_score'], reverse=True)
top_chunks = reranked_chunks[:5]
candidate_emb = [tc['embeddings'] for tc in reranked_chunks]
for c in candidate_emb:
    print(len(query_embedding), len(c))
    print(cosine_similarity(query_embedding, c))
print(f'Rerank: {rerank_scores}')


# Image/Video
# candidate_emb = search_results.get('embeddings')[0]
# for c in candidate_emb:
#     print(len(query_embedding), len(c))
#     print(cosine_similarity(query_embedding, c))

# Pair compare
# candidate_emb = collection.get(ids=["CCTV4|VID_GEN1_chunk_0.mp4_text", "CCTV4|VID_GEN1_chunk_0.mp4_image"],
#                         include=['embeddings']).get('embeddings')
# print(cosine_similarity(candidate_emb[0], candidate_emb[1]))

