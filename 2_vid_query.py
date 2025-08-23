import torch
import chromadb
from transformers import AutoProcessor, AutoModel
from FlagEmbedding import FlagReranker

# --- Config ---
CHROMA_PATH = "./out/chroma_db"
COLLECTION_NAME = "video_embeddings"
MODEL_ID = "microsoft/xclip-base-patch16-16-frames"
RERANKER_MODEL_NAME = "BAAI/bge-reranker-base"

# --- Device selection (MPS on Mac) ---
if torch.backends.mps.is_available():
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
print(f"Using device: {device}")

# --- Load X-CLIP model (same as used for video encoding) ---
processor = AutoProcessor.from_pretrained(MODEL_ID)
model = AutoModel.from_pretrained(MODEL_ID).to(device)
model.eval()

# --- Load reranker (cross-encoder) ---
reranker = FlagReranker(RERANKER_MODEL_NAME)

# --- Connect to Chroma ---
client = chromadb.PersistentClient(path=CHROMA_PATH)
collection = client.get_or_create_collection(name=COLLECTION_NAME)

# --- Text embedding function ---
def get_text_embedding(text, device="cpu"):
    inputs = processor(text=[text], return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        emb = model.get_text_features(**inputs)
        emb = emb / emb.norm(dim=-1, keepdim=True)
    return emb[0].cpu().numpy()

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