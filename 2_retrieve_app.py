import streamlit as st
import chromadb
import os
from sentence_transformers import SentenceTransformer, CrossEncoder
import cv2
import base64
import json, ast
from FlagEmbedding import FlagReranker

# --- Configuration ---
DATA_DIR = "./data/videos"
OUTPUT_CHROMA_DIR = "./out/chroma_db"
COLLECTION_NAME = "chunk_captions"
EMBEDDING_MODEL_NAME = 'BAAI/bge-m3'
RERANKER_MODEL_NAME = 'BAAI/bge-reranker-large'
# RERANKER_MODEL_NAME = 'BAAI/bge-reranker-v2-gemma'
ANNOTATED_CHUNK_DIR = "./out/video_chunks_cv"
OUTPUT_METADATA_DIR = "./out/metadata"
VIDEO_COLLECTION_NAME = "video_captions"

# --- Caching Models for Performance ---
@st.cache_resource
def load_models():
    embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    reranker_model = FlagReranker(RERANKER_MODEL_NAME, use_fp16=True)
    print("Models loaded successfully.")
    return embedding_model, reranker_model

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

def search_and_rerank(query, embedding_model, reranker_model, collection, top_n=25, rerank_top_k=5):
    if not query:
        return []
    query_embedding = embedding_model.encode(query, normalize_embeddings=True).tolist()
    initial_results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_n,
        include=["metadatas", "documents", "embeddings", "distances"]
    )
    if not initial_results['ids'][0]:
        return []

    rerank_pairs = []
    for metadata in initial_results['metadatas'][0]:
        rerank_pairs.append([query, metadata.get('caption', '')])
    rerank_scores = reranker_model.compute_score(rerank_pairs)
    reranked_chunks = []
    for i, meta in enumerate(initial_results['metadatas'][0]):
        reranked_chunks.append({
            'metadata': meta,
            'rerank_score': rerank_scores[i]
        })
    reranked_chunks.sort(key=lambda x: x['rerank_score'], reverse=True)
    top_chunks = reranked_chunks[:rerank_top_k]
    # return top_chunks
    video_results = {}
    for chunk in top_chunks:
        chunk_path = chunk['metadata'].get('chunk_path', '')
        video_name = '.'.join([os.path.basename(chunk_path).split('_chunk_')[0],os.path.basename(chunk_path).split('.')[-1]])
        video_path = os.path.join(DATA_DIR, video_name)
        if video_name not in video_results:
            video_results[video_name] = {
                'video_name': video_name,
                'best_score': chunk['rerank_score'],
                'matching_chunks': [chunk]}
        else:
            video_results[video_name]['matching_chunks'].append(chunk)
            if chunk['rerank_score'] > video_results[video_name]['best_score']:
                video_results[video_name]['best_score'] = chunk['rerank_score']
    final_list = sorted(video_results.values(), key=lambda x: x['best_score'], reverse=True)
    return final_list

def browse_collection(collection, limit=30):
    if not collection:
        return []
    results = collection.get(
        limit=limit,
        include=["metadatas"]
    )
    if not results['ids']:
        return []
    formatted_results = []
    for meta in results['metadatas']:
        formatted_results.append({'metadata': meta})
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
            caption = parse_json(caption_str)
            st.subheader(f"Rank {i+1}")
            if isinstance(caption, dict) and caption:
                st.markdown(f"**Video Summary:** {caption.get('Video Summary', '')}")
                st.markdown(f"**Features:**")
                for feature in caption.get("Features", []):
                    st.text(f"- {' | '.join(str(v) for v in feature.values())}")
            else:
                st.markdown(f"**Caption:** `{caption_str}`")
            if 'rerank_score' in result:
                score = result['rerank_score']
                color = "green" if score > 0 else "red"
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
        score_color = "green" if best_score > 0 else "red"
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
            st.markdown(f"**Score:** <span style='color:{'green' if best_score>0 else 'red'}; font-weight:bold;'>{best_score:.4f}</span>", unsafe_allow_html=True)
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
                for other_chunk in all_matching_chunks[:]:
                    score = other_chunk['rerank_score']
                    path = other_chunk['metadata'].get('chunk_path', 'N/A')
                    summary = parse_json(other_chunk['metadata'].get('caption', '{}')).get('Video Summary', 'N/A')
                    col1, col2 = st.columns([1, 2])
                    with col1:
                        video_path = best_chunk['metadata'].get('chunk_path', '')
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
                        st.markdown(f"**Score:** <span style='color:{'green' if score>0 else 'red'}; font-weight:bold;'>{score:.4f}</span>", unsafe_allow_html=True)
                        st.markdown(f"**Chunk Path:** `{best_chunk['metadata'].get('chunk_path', 'N/A')}`")
                        # st.markdown(f"**Chunk:** `{os.path.basename(path)}` | "
                        #         f"**Score:** <span style='color:{score_color}; font-weight:bold;'>{best_score:.4f}</span> | "
                        #         f"**Summary:** {summary}", unsafe_allow_html=True)
                        

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
embedding_model, reranker_model = load_models()
collection, video_collection = load_chroma_collection()

# --- Create Tabs ---
tab1, tab2, tab3 = _col2.tabs(["🔎 Search Chunk by Query", "📚 Browse Chunks", "📜 Browse Video Summaries"])

# --- Search Tab ---
with tab1:
    query = st.text_input("Enter your search query:", "Group of male and female subjects walking together. One of them wearing a T-shirt with strawberry motifs.")
    if st.button("Search"):
        with st.spinner("Searching and reranking..."):
            search_results = search_and_rerank(query, embedding_model, reranker_model, collection)
            # display_results(search_results)
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