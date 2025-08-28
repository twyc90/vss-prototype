import streamlit as st
import chromadb
import os
from sentence_transformers import SentenceTransformer, CrossEncoder
import cv2
import base64
import json, ast
from FlagEmbedding import FlagReranker
from PIL import Image
from transformers import AutoTokenizer, AutoProcessor, AutoModel, XCLIPModel
import torch
import numpy as np
from PIL import Image
import tempfile

# --------------
# --- CONFIG ---
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

# ------------------------
# --- Emb Model Config ---
# ------------------------
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


# --------------------------------------
# --- Caching Models for Performance ---
# --------------------------------------
@st.cache_resource
def load_models():
    # embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    processor = AutoProcessor.from_pretrained(EMBEDDING_MODEL_NAME)
    embedding_model = AutoModel.from_pretrained(EMBEDDING_MODEL_NAME).to(device)
    embedding_model.eval()
    reranker_model = FlagReranker(RERANKER_MODEL_NAME, use_fp16=True)
    print("Models loaded successfully.")
    return embedding_model, reranker_model, processor

@st.cache_resource
def load_chroma_collection():
    if not os.path.exists(OUTPUT_CHROMA_DIR):
        st.error(f"ChromaDB directory not found at '{OUTPUT_CHROMA_DIR}'. Please run the processing script first.")
        return None
    client = chromadb.PersistentClient(path=OUTPUT_CHROMA_DIR)
    collection = client.get_collection(name=COLLECTION_NAME)
    video_collection = client.get_collection(name=VIDEO_COLLECTION_NAME)
    print("Collection loaded successfully.")
    return collection, video_collection


def search_and_rerank(query_text=None, query_image=None, query_video=None, processor=None, embedding_model=None, reranker_model=None, collection=None, top_n=50, rerank_top_k=10, min_score=0.6):
    if query_text is not None:
        type='text'
    elif query_image is not None:
        type='image'
    elif query_video is not None:
        type='image'

    query_embedding = []
    if query_text:
        emb_text = encode_text([query_text], processor, embedding_model, device=device)[0]
        query_embedding = emb_text
    elif query_image:
        img = Image.open(query_image)
        emb_image = encode_image([img], processor, embedding_model, device=device)[0]
        query_embedding = emb_image
    elif query_video:
        cap = cv2.VideoCapture(query_video)
        fps = cap.get(cv2.CAP_PROP_FPS)
        all_frames = []
        while True:
            ret, frame = cap.read()
            if not ret: break
            all_frames.append(frame)
        cap.release()
        if not all_frames: 
            return "No frames extracted for captioning."
        num_frames = len(all_frames)
        indices = np.linspace(0, num_frames - 1, num=min(num_frames, 8), dtype=int)
        sampled_frames = [all_frames[i] for i in indices]
        pil_frames = [Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)) for frame in sampled_frames]
        query_embedding = encode_video(pil_frames, processor, embedding_model, device=device)

    search_results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_n,
        include=['metadatas','documents','embeddings','distances'],
        # where={'type':type}
    )
    if type=='text':
        rerank_pairs = []
        for metadata in search_results['metadatas'][0]:
            rerank_pairs.append([query_text, metadata.get('caption', '')])
        rerank_scores = reranker_model.compute_score(rerank_pairs)
        rerank_scores = torch.sigmoid(torch.tensor(rerank_scores)).tolist()
        reranked_chunks = []
        for i, meta in enumerate(search_results['metadatas'][0]):
            reranked_chunks.append({
                'metadata': meta,
                'rerank_score': rerank_scores[i]
            })
        reranked_chunks.sort(key=lambda x: x['rerank_score'], reverse=True)
        top_chunks = reranked_chunks
    elif type=='image':
        matched_chunks = []
        for meta, distance in zip(search_results['metadatas'][0], search_results['distances'][0]):
            matched_chunks.append({
                'metadata': meta,
                'rerank_score': 1-distance
            })
        matched_chunks.sort(key=lambda x: x['rerank_score'], reverse=True)
        top_chunks = matched_chunks

    video_results = {}
    for chunk in top_chunks:
        chunk_path = chunk['metadata'].get('chunk_path', '')
        video_name = '.'.join([os.path.basename(chunk_path).split('_chunk_')[0],os.path.basename(chunk_path).split('.')[-1]])
        video_path = os.path.join(DATA_DIR, video_name)
        if video_name not in video_results:
            video_results[video_name] = {
                'video_name': video_name,
                'best_score': chunk['rerank_score'],
                'matching_chunks': [chunk],
                'chunk_paths': [chunk_path]
                }
        else:
            if chunk_path not in video_results[video_name]['chunk_paths']:
                video_results[video_name]['matching_chunks'].append(chunk)
                video_results[video_name]['chunk_paths'].append(chunk_path)
            if chunk['rerank_score'] > video_results[video_name]['best_score']:
                video_results[video_name]['best_score'] = chunk['rerank_score']
    final_list = sorted(video_results.values(), key=lambda x: x['best_score'], reverse=True)
    final_list = sorted([v for v in video_results.values() if v['best_score']>=min_score], key=lambda x: x['best_score'], reverse=True)
    return final_list[:rerank_top_k]


def browse_collection(collection, limit=30):
    if not collection:
        return []
    results = collection.get(
        limit=limit,
        include=["metadatas","embeddings"]
    )
    formatted_results = []
    # ids = set()
    for id, meta, emb in zip(results['ids'], results['metadatas'], results['embeddings']):
        # if meta['video_name'] not in ids:
        formatted_results.append({'metadata': meta, 'embedding':emb})
            # ids.add(meta['video_name'])
    return formatted_results


def parse_json(json_string):
    json_string = json_string.replace('```json','').replace('```','')
    try:
        json_obj = json.loads(json_string)
    except Exception as e:
        try:
            json_obj = ast.literal_eval(json_string)
        except Exception as e:
            json_obj = {}
    return json_obj


def display_results(results_list):
    if not results_list:
        st.warning("No results found.")
        return
    st.success(f"Displaying {len(results_list)} results:")
    for i, result in enumerate(results_list):
        st.markdown("---")
        _col1, _col2, _col3 = st.columns([1, 10, 1])
        col1, col2 = _col2.columns([1, 2])
        with col1:
            video_path = result['metadata'].get('chunk_path', '')
            if os.path.exists(video_path):
                st.video(video_path)
            else:
                st.warning(f"Video not found at: {video_path}")
        with col2:
            caption_str = result["metadata"].get('caption', '{}')
            type = result["metadata"].get('type','')
            emb = result["embedding"]
            caption = parse_json(caption_str)
            st.subheader(f"Rank {i+1}")
            if isinstance(caption, dict) and caption:
                st.markdown(f"**Video Summary:** {caption.get('Video Summary', '')}")
                st.markdown(f"**Features:**")
                for feature in caption.get("Features", []):
                    st.text(f"- {' | '.join(str(v) for v in feature.values())}")
                st.markdown(f"**Source**: {type}")
                with st.expander("Embeddings"):
                    st.text(f"{', '.join([str(e) for e in emb.tolist()])}")
            else:
                st.markdown(f"**Caption:** `{caption_str}`")
            if 'rerank_score' in result:
                score = result['rerank_score']
                color = "green" if score>COLOR_SCORE else "red"
                st.markdown(
                    f"""<p><strong>Re-rank Score: </strong>
                        <span style='color:{color}; font-weight:bold;'>{score:.4f}</span>
                    </p>""",
                    unsafe_allow_html=True)
            st.markdown(f"**Chunk Path:** `{result['metadata'].get('chunk_path', 'N/A')}`")
            chunk_path = result['metadata'].get('chunk_path', 'N/A')
            if chunk_path:
                metadata_filename = os.path.basename(chunk_path).replace(os.path.splitext(chunk_path)[1], '.json')
                metadata_path = os.path.join(OUTPUT_METADATA_DIR, metadata_filename)
                if os.path.exists(metadata_path):
                    with st.expander("Show Object Tracking Info"):
                        try:
                            with open(metadata_path, 'r') as f:
                                metadata = json.load(f)
                            objects = metadata.get("objects", [])
                            if objects:
                                for obj in objects:
                                    label = obj.get('label', 'N/A')
                                    confidence = obj.get('last_confidence') 
                                    st.markdown(f"**Object ID {obj.get('id', 'N/A')}**: `{label}` (Confidence: `{confidence:.2f}`)")
                            else:
                                st.write("No objects were tracked in this video chunk.")
                        except Exception as e:
                            st.error(f"Could not read metadata file: {e}")


def display_summary_results(results_list):
    if not results_list:
        st.warning("No video summaries found in the collection.")
        return
    st.success(f"Displaying {len(results_list)} random video summaries:")
    for result in results_list:
        st.markdown("---")
        _col1, _col2, _col3 = st.columns([1, 10, 1])
        col1, col2 = _col2.columns([1, 2])
        video_name = result['metadata'].get('video_name', 'N/A')
        summary = result['metadata'].get('summary', 'No summary available.')
        video_path = result['metadata'].get('video_path', '')
        with col1:
            if os.path.exists(video_path):
                st.video(video_path, width=300)
            else:
                st.warning(f"Video not found at: {video_path}")
        with col2:
            st.subheader(f"🎬 Summary for: `{video_name}`")
            st.markdown(summary)


def display_video_results(results_list):
    if not results_list:
        st.warning("No results found.")
        return
    st.success(f"Found {len(results_list)} matching videos:")
    for i, result in enumerate(results_list):
        video_name = result['video_name']
        best_score = result['best_score']
        all_matching_chunks = result['matching_chunks']
        all_matching_chunks.sort(key=lambda x: x['rerank_score'], reverse=True)
        best_chunk = all_matching_chunks[0]
        st.markdown("---")
        score_color = "green" if best_score > COLOR_SCORE else "red"
        st.header(f"Rank {i+1}: {video_name}")
        st.markdown(f"**Relevance Score:** <span style='color:{score_color}; font-weight:bold;'>{best_score:.4f}</span>", unsafe_allow_html=True)
        st.info(f"Found {len(all_matching_chunks)} relevant chunk(s) in this video.")

        st.subheader("Best Matching Chunk (Preview)")
        col1, col2 = st.columns([1, 2])
        with col1:
            video_path = best_chunk['metadata'].get('chunk_path', '')
            if os.path.exists(video_path):
                st.video(video_path)
            else:
                st.warning(f"Video not found: {video_path}")
        with col2:
            caption_str = best_chunk["metadata"].get('caption', '{}')
            caption = parse_json(caption_str)
            if isinstance(caption, dict) and caption:
                st.markdown(f"**Summary:** {caption.get('Video Summary', '')}")
                st.markdown(f"**Features:**")
                for feature in caption.get("Features", []):
                    st.text(f"- {' | '.join(str(v) for v in feature.values())}")
            else:
                st.markdown(f"**Caption:** `{caption_str}`")
            st.markdown(f"**Score:** <span style='color:{'green' if best_score>COLOR_SCORE else 'red'}; font-weight:bold;'>{best_score:.4f}</span>", unsafe_allow_html=True)
            st.markdown(f"**Chunk Path:** `{best_chunk['metadata'].get('chunk_path', 'N/A')}`")
            chunk_path = best_chunk['metadata'].get('chunk_path', 'N/A')
            if chunk_path:
                metadata_filename = os.path.basename(chunk_path).replace(os.path.splitext(chunk_path)[1], '.json')
                metadata_path = os.path.join(OUTPUT_METADATA_DIR, metadata_filename)
                if os.path.exists(metadata_path):
                    with st.expander("Show Object Tracking Info"):
                        try:
                            with open(metadata_path, 'r') as f:
                                metadata = json.load(f)
                            objects = metadata.get("objects", [])
                            if objects:
                                for obj in objects:
                                    label = obj.get('label', 'N/A')
                                    confidence = obj.get('last_confidence') 
                                    st.markdown(f"**Object ID {obj.get('id', 'N/A')}**: `{label}` (Confidence: `{confidence:.2f}`)")
                            else:
                                st.write("No objects were tracked in this video chunk.")
                        except Exception as e:
                            st.error(f"Could not read metadata file: {e}")

        if len(all_matching_chunks) > 0:
            with st.expander("View all relevant chunks in this video"):
                for other_chunk in all_matching_chunks:
                    score = other_chunk['rerank_score']
                    path = other_chunk['metadata'].get('chunk_path', 'N/A')
                    summary = parse_json(other_chunk['metadata'].get('caption', '{}')).get('Video Summary', 'N/A')
                    col1, col2 = st.columns([1, 2])
                    with col1:
                        video_path = other_chunk['metadata'].get('chunk_path', '')
                        if os.path.exists(video_path):
                            st.video(video_path)
                        else:
                            st.warning(f"Video not found: {video_path}")
                    with col2:
                        caption_str = other_chunk["metadata"].get('caption', '{}')
                        caption = parse_json(caption_str)
                        if isinstance(caption, dict) and caption:
                            st.markdown(f"**Summary:** {caption.get('Video Summary', '')}")
                        else:
                            st.markdown(f"**Caption:** `{caption_str}`")
                        st.markdown(f"**Score:** <span style='color:{'green' if score>COLOR_SCORE else 'red'}; font-weight:bold;'>{score:.4f}</span>", unsafe_allow_html=True)
                        st.markdown(f"**Chunk Path:** `{best_chunk['metadata'].get('chunk_path', 'N/A')}`")


# --- Streamlit UI ---
st.set_page_config(layout="wide")
_col1, _col2, _col3 = st.columns([1, 10, 1])
col1, col2 = _col2.columns([1, 2])
_col2.title("🎬 Video Semantic Search")
with _col2:
    st.write("") # Spacer for alignment
    if st.button("🔄 Refresh Collection"):
        st.cache_resource.clear()
        st.success("Cache cleared! Reloading collection...")
        st.rerun()


# Load models and collection
embedding_model, reranker_model, processor = load_models()
collection, video_collection = load_chroma_collection()

# --- Create Tabs ---
tab1, tab2, tab3 = _col2.tabs(["🔎 Search Chunk by Query", "📚 Browse Chunks", "📜 Browse Video Summaries"])

# --- Search Tab ---
with tab1:
    with st.expander("Search Parameters"):
        param_col11, param_col21 = st.columns([5,10])
        param_col12, param_col22 = st.columns([5,10])
        with param_col11:
            min_score_val = st.slider(
                "Minimum Score Threshold",
                min_value=0.0,
                max_value=1.0,
                step=0.01,
                value=MIN_SCORE,
                help="Filters out results with a relevance score below this value."
            )
        with param_col12:
            reranker_top_k_val = st.slider(
                "Number of Results to Display (Top K)",
                min_value=1,
                max_value=20,
                step=1,
                value=RERANKER_TOP_K,
                help="The maximum number of final video results to show."
            )

    query_mode = st.radio("Choose query type:", ["text", "image", "video"])
    query_text, query_image, query_video = None, None, None

    if query_mode in ["text"]:
        query_text = st.text_input("Enter your search query:", "Group of male and female subjects walking together. One of them wearing a T-shirt with strawberry motifs.")

    elif query_mode in ["image"]:
        uploaded_file = st.file_uploader("Upload an image:", type=["jpg", "png", "jpeg"])
        if uploaded_file is not None:
            query_image = uploaded_file

    elif query_mode in ["video"]:
        uploaded_file = st.file_uploader("Upload an video:", type=["mp4"])
        if uploaded_file is not None:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
                temp_file.write(uploaded_file.read())
                query_video = temp_file.name
            
    if st.button("Search"):
        with st.spinner("Searching and reranking..."):
            search_results = search_and_rerank(query_text, query_image, query_video, processor, embedding_model, reranker_model, collection, rerank_top_k=reranker_top_k_val, min_score=min_score_val)
            display_video_results(search_results)

# --- Browse Chunk Tab ---
with tab2:
    st.subheader(f"Browsing the random 30 items in the collection")
    if st.button("Load Items"):
        with st.spinner("Fetching collection..."):
            browse_results = browse_collection(collection, limit=30)
            display_results(browse_results)


# --- Browse Video Summaries Tab ---
with tab3:
    st.subheader("Browse random aggregated video summaries")
    if st.button("Load Random Summaries"):
        with st.spinner("Fetching summaries..."):
            summary_results = browse_collection (video_collection, limit=10)
            display_summary_results(summary_results)