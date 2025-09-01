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
from collections import defaultdict
from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility
from pymilvus import MilvusClient

# --------------
# --- Config ---
# --------------
DATA_DIR = "./data/videos"
OUTPUT_DIR = "./out"
OUTPUT_CHUNK_DIR = "./out/video_chunks"
ANNOTATED_CHUNK_DIR = "./out/video_chunks_cv"
CV_CHUNK_DIR = "./out/video_chunks_cv/obj"
OUTPUT_METADATA_DIR = "./out/metadata"
OUTPUT_CHROMA_DIR = "./out/chroma_db"
SAVE_CHUNKS = True
CHUNK_SIZE = 10           # in seconds
FRAMES_PER_CHUNK = 8     # frames per chunk
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(OUTPUT_CHUNK_DIR, exist_ok=True)
os.makedirs(ANNOTATED_CHUNK_DIR, exist_ok=True)
os.makedirs(CV_CHUNK_DIR, exist_ok=True)
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

# ------------------------
# --- Emb Model Config ---
# ------------------------
# EMBEDDING_MODEL_NAME = 'BAAI/bge-m3'
# embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
EMBEDDING_MODEL_NAME = 'BAAI/BGE-VL-base'
processor = AutoProcessor.from_pretrained(EMBEDDING_MODEL_NAME)
embedding_model = AutoModel.from_pretrained(EMBEDDING_MODEL_NAME).to(device)

# -------------------------
# --- Connect to Chroma ---
# -------------------------
COLLECTION_NAME = "chunk_captions"
client = chromadb.PersistentClient(path=OUTPUT_CHROMA_DIR)
collection = client.get_or_create_collection(name=COLLECTION_NAME, metadata={"hnsw:space": "cosine"})
VIDEO_COLLECTION_NAME = "video_captions"
video_collection = client.get_or_create_collection(name=VIDEO_COLLECTION_NAME, metadata={"hnsw:space": "cosine"})

# ------------------------
# --- Milvus Lite Setup ---
# ------------------------
MILVUS_DB_PATH = "./out/milvus.db"
CHUNK_COLLECTION = "chunk_captions"
VIDEO_COLLECTION = "video_captions"
connections.connect("default", uri=MILVUS_DB_PATH)
with torch.no_grad():
    dummy = processor(text="hello", return_tensors="pt").to(device)
    dim = embedding_model.get_text_features(**dummy).shape[-1]
print(f'Embedding size: {dim}')

index_params = {
    "metric_type": "IP",
    "index_type": "IVF_FLAT",
    "params": {"nlist": 128}
}
chunk_fields = [
    # FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
    FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, auto_id=False, max_length=512),
    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dim),
    FieldSchema(name="metadata", dtype=DataType.JSON),
]
video_fields = [
    # FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
    FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, auto_id=False, max_length=512),
    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dim),
    FieldSchema(name="metadata", dtype=DataType.JSON),
]
chunk_schema = CollectionSchema(chunk_fields, description="video search collection", enable_dynamic_field=False)
video_schema = CollectionSchema(video_fields, description="summary search collection", enable_dynamic_field=False)
collection = Collection(name=CHUNK_COLLECTION, schema=chunk_schema, using='default')
collection.create_index(field_name="embedding", index_params=index_params)
video_collection = Collection(name=VIDEO_COLLECTION, schema=video_schema, using='default')
video_collection.create_index(field_name="embedding", index_params=index_params)
db_type='milvus'

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
VLM_MODEL_NAME = "google/gemma-3-12b"
# VLM_MODEL_NAME = "google/gemma-3-27b"
LLM_MODEL_NAME = "qwen/qwen2.5-vl-7b"
# client = OpenAI(base_url=VLM_API_BASE, api_key=VLM_API_KEY)
client = AsyncOpenAI(base_url=VLM_API_BASE, api_key=VLM_API_KEY)

# ------------------------
# --- Helper Functions ---
# ------------------------
def encode_text(texts, processor, model, device="cpu", normalize=True, max_length=77):
    embeddings = []
    try:
        for text in texts:
            tokens = processor(text=text, return_tensors="pt", add_special_tokens=True)
            input_ids = tokens["input_ids"][0]
            if len(input_ids) <= max_length:
                # normal encode
                inputs = processor(
                    text=text,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=max_length
                ).to(device)
                with torch.no_grad():
                    outputs = model.get_text_features(**inputs)
                emb = outputs.cpu().numpy()
            else:
                # chunk and mean encode
                chunks = [input_ids[i:i+max_length] for i in range(0, len(input_ids), max_length)]
                chunk_embs = []
                for chunk in chunks:
                    inputs = {"input_ids": chunk.unsqueeze(0).to(device)}
                    with torch.no_grad():
                        outputs = model.get_text_features(**inputs)
                    chunk_embs.append(outputs.cpu().numpy())
                emb = np.mean(chunk_embs, axis=0)
            # normalize
            if normalize:
                if emb.ndim == 1:
                    emb = emb / np.linalg.norm(emb)
                else:
                    emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)
            embeddings.append(emb.tolist())
    except Exception as e:
        print(f"Encoding error: {e}")
        return None
    return embeddings

# def encode_text(texts, processor, model, device="cpu", normalize=True, max_length=77):
#     embeddings = []
#     try:
#         for text in texts:
#             tokens = processor(text=text, return_tensors="pt", add_special_tokens=True)
#             input_ids = tokens["input_ids"][0]
#             if len(input_ids) <= max_length:
#                 print('normal encode')
#                 inputs = processor(text=text, return_tensors="pt", padding=True, truncation=True, max_length=max_length).to(device)
#                 with torch.no_grad():
#                     outputs = model.get_text_features(**inputs)
#                 emb = outputs.cpu().numpy()
#             else:
#                 print('chunk and mean encode')
#                 chunks = [input_ids[i:i+max_length] for i in range(0, len(input_ids), max_length)]
#                 chunk_embs = []
#                 for chunk in chunks:
#                     inputs = {"input_ids": chunk.unsqueeze(0).to(device)}
#                     with torch.no_grad():
#                         outputs = model.get_text_features(**inputs)
#                     chunk_embs.append(outputs.cpu().numpy())
#                 emb = np.mean(chunk_embs, axis=0)
#             if normalize:
#                 emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)
#     except Exception as e:
#         print(f'Encoding error. {e}')
#     return emb.tolist()

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


def chunk_video(video_path):
    print(f"Chunking video: {video_path}")
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    chunk_frames = int(fps * CHUNK_SIZE)
    chunk_count = 0
    video_filename = os.path.basename(video_path)
    video_name, video_ext = os.path.splitext(video_filename)
    chunk_paths = []
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
        # indices = np.linspace(0, len(frames) - 1, num=FRAMES_PER_CHUNK, dtype=int)
        # frames = [frames[i] for i in indices]
        chunk_path = os.path.join(OUTPUT_CHUNK_DIR, chunk_filename)
        chunk_paths.append(chunk_path)
        fourcc = cv2.VideoWriter_fourcc(*'avc1')
        out = cv2.VideoWriter(chunk_path, fourcc, fps, (frames[0].shape[1], frames[0].shape[0]))
        for frame in frames:
            out.write(frame)
        out.release()
        chunk_count += 1
    cap.release()
    return chunk_paths

def has_motion_diff(video_path, threshold=30, min_motion_ratio=0.02):
    cap = cv2.VideoCapture(video_path)
    ret, prev = cap.read()
    if not ret:
        return False
    prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
    motion_pixels = 0
    total_pixels = prev_gray.size
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        diff = cv2.absdiff(prev_gray, gray)
        _, diff = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)

        motion_pixels += np.sum(diff > 0)
        frame_count += 1
        prev_gray = gray
    cap.release()
    if frame_count == 0:
        return False
    motion_ratio = motion_pixels / (frame_count * total_pixels)
    return motion_ratio > min_motion_ratio

def has_motion_bgsub(video_path, min_motion_frames=5):
    cap = cv2.VideoCapture(video_path)
    fgbg = cv2.createBackgroundSubtractorMOG2()
    motion_frames = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        fgmask = fgbg.apply(frame)
        if cv2.countNonZero(fgmask) > 500:
            motion_frames += 1
    cap.release()
    return motion_frames >= min_motion_frames


def detect_and_track_objects(chunk_paths):
    chunk_cv_paths = []
    for chunk_path in chunk_paths:
        print(f"Detect motion & track object: {chunk_path}")
        cap = cv2.VideoCapture(chunk_path)

        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        output_video_path = os.path.join(ANNOTATED_CHUNK_DIR, os.path.basename(chunk_path))
        fourcc = cv2.VideoWriter_fourcc(*'avc1')
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
        is_motion_diff, is_motion_bgsub = has_motion_diff(chunk_path), has_motion_bgsub(chunk_path)
        has_motion = is_motion_diff or is_motion_bgsub
        chunk_metadata = {
            "chunk_path": chunk_path,
            "motion_detected": str(has_motion),
            "objects": []
        }
        os.remove(chunk_path)
        if not has_motion:
            print(f"No motion detected in {chunk_path}. Skipping object tracking.")
            cap.release()
            out.release()
            os.remove(output_video_path)
            continue

        chunk_cv_paths.append(output_video_path)
        tracked_objects = {}
        next_object_id = 0
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            latest_boxes_to_draw = []
            results = model(frame, verbose=False)
            for result in results:
                for box in result.boxes:
                    if box.conf[0] > CONFIDENCE_THRESHOLD:
                        # coords_float = box.xyxy[0].cpu().numpy().tolist()
                        coords = box.xyxy[0].cpu().numpy().astype(int).tolist()
                        class_id = int(box.cls[0].cpu().numpy())
                        class_name = model.names[class_id]
                        center_x = (coords[0] + coords[2]) / 2
                        center_y = (coords[1] + coords[3]) / 2
                        confidence = box.conf[0].cpu().numpy().item()
                        found_match = False
                        display_id = -1
                        for obj_id, obj_data in tracked_objects.items():
                            dist = np.sqrt((center_x - obj_data['last_pos'][0])**2 + (center_y - obj_data['last_pos'][1])**2)
                            if obj_data['label'] == class_name and dist < 75:
                                obj_data['last_pos'] = (center_x, center_y)
                                # if frame_idx not in obj_data['frames_present']:
                                #     obj_data['frames_present'].append(frame_idx)
                                #     obj_data['obj_bbox'][frame_idx] = {frame_idx: coords}
                                # update best bbox if confidence higher
                                if confidence > obj_data['best_confidence']:
                                    obj_data['best_frame_idx'] = frame_idx
                                    obj_data['best_confidence'] = confidence
                                    obj_data['best_bbox'] = coords
                                    obj_data['best_frame'] = frame.copy()
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
                                # 'frames_present': [frame_idx],
                                # 'obj_bbox': {frame_idx: coords},
                                # 'last_confidence': confidence,
                                'best_frame_idx': frame_idx,
                                'best_bbox': coords,
                                'best_confidence': confidence,
                                'best_frame': frame.copy()
                            }
                            next_object_id += 1
                        label = f"ID {display_id}: {class_name} {confidence:.3f}"
                        latest_boxes_to_draw.append((coords, label))
            for coords, label in latest_boxes_to_draw:
                x1, y1, x2, y2 = coords
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            out.write(frame)
            frame_idx += 1
        # save crops of best bboxes
        for obj in tracked_objects.values():
            x1, y1, x2, y2 = obj['best_bbox']
            frame = obj['best_frame']
            label = obj['label']
            conf = obj['best_confidence']
            # crop region
            crop = frame[y1:y2, x1:x2].copy()
            # draw bbox (relative to crop)
            h, w = crop.shape[:2]
            cv2.rectangle(crop, (0, 0), (w - 1, h - 1), (0, 255, 0), 2)
            # put label + confidence
            text = f"{label} {conf:.2f}"
            cv2.putText(crop, text, (5, 20), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 255, 0), 2, lineType=cv2.LINE_AA)
            # save annotated crop
            crop_path = os.path.join(CV_CHUNK_DIR, f"{os.path.basename(chunk_path)}_obj{obj['id']}.jpg")
            cv2.imwrite(crop_path, crop)
            obj['crop_path'] = crop_path
            obj.pop("best_frame", None)
        chunk_metadata["objects"] = list(tracked_objects.values())
        metadata_filename = os.path.basename(chunk_path).replace(os.path.splitext(chunk_path)[1], '.json')
        metadata_path = os.path.join(OUTPUT_METADATA_DIR, metadata_filename)
        with open(metadata_path, 'w') as f:
            json.dump(chunk_metadata, f, indent=4)
        cap.release()
        out.release()
    return chunk_cv_paths


async def vlm_caption(chunk_path):
    print(f"Generating caption for: {chunk_path}")
    cap = cv2.VideoCapture(chunk_path)
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
    indices = np.linspace(0, num_frames - 1, num=min(num_frames, FRAMES_PER_CHUNK), dtype=int)
    sampled_frames = [all_frames[i] for i in indices]
    base64_frames = []
    for frame in sampled_frames:
        _, buffer = cv2.imencode(".jpg", frame)
        base64_frames.append(base64.b64encode(buffer).decode("utf-8"))
    try:
        cv_metadata = []
        chunk_filename = chunk_path.split(ANNOTATED_CHUNK_DIR+'/')[1]
        metadata_filename = chunk_filename.replace(os.path.splitext(chunk_filename)[1], '.json')
        metadata_path = os.path.join(OUTPUT_METADATA_DIR, metadata_filename)
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        objects = metadata.get("objects", [])
        if objects:
            for obj in objects:
                label = obj.get('label', 'N/A')
                confidence = obj.get('best_confidence')
                cv_metadata.append(f"Object ID {obj.get('id', 'N/A')}: {label}, Best Confidence: {confidence:.3f}")
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": (
        "These are sequential frames from a video clip.\n"
        "Objects detected are bounded by a green color box, top left corner of each box shows the object ID, object label and object confidence.\n"
        "You are now expert investigation officer that looks for distinguished and/or suspicious feature from video clips for national security concerns.\n"
        "1. Identify all the human objects detected and return their features such as clothing, appearance, etc and deduce what they are doing.\n"
        "2. Identify all the nonhuman objects like car, personal mobility devices (PMD), bicycle, scooter, etc and record their color and features.\n"
        "3. Identify all the suspicious objects like unclaimed bag in the middle of a bus stop that looked suspicious, etc and provide the description\n"
        "5. Identify the likely location of the video clip, such as bus stop, shopping mall, restaurant, MRT, MBS, etc\n"
        "6. Generate a concise one sentence caption for the video clip.\n"
        "7. Return your result in a valid JSON output as per below\n"
        "Example : "
        "        ------CHUNK DATA START HERE------\n"
        "        CV METADATA:\n        "
        "        Object ID 0: person, Best Confidence: 0.851"
        "        Object ID 1: bus, Best Confidence: 0.523"
        "        Object ID 2: car, Best Confidence: 0.995"
        "        Object ID 3: person, Best Confidence: 0.926"
        "Output : "
        '''
        {
        "Features": [
        {
        "description": "A young man wearing a backpack, jeans, and a light-colored shirt. He appears to be looking at his phone. There is a red color handbag on the floor near to him.",
        "tracker": "Young man",
        "location": "Standing near the edge of a bus stop shelter.",
        },
        {
        ""description": "A green color SG bus with number 298",
        "tracker": "Bus",
        "location": "On the road",
        },
        {
        ""description": "A bright yellow color Honda civic",
        "tracker": "Car",
        "location": "On the road",
        },
        {
        ""description": "An elderly woman wearing a dark jacket and seated on the bus stop bench.",
        "tracker": "Elderly Woman",
        "location": "Seated on the bus stop bench."
        }],
        "Video Summary": 
        "this video clip is in a restaurant setting, there are two individual in the video."
        }\n'''
        "        ------CHUNK DATA START HERE------\n"
        "        CV METADATA:\n        "
        f"{'\n        '.join(cv_metadata)}"
        "\nOutput : "

                    )},
                    *[{"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_str}"}} for img_str in base64_frames]
                ]
            },
        ]
        # print(messages[0]['content'][0]['text'])
        response = await client.chat.completions.create(
            model=VLM_MODEL_NAME,
            messages=messages,
            max_tokens=2048,
        )
        caption = response.choices[0].message.content
        # print(f"-"*25)
        # print(f"Generated Caption: {caption}")
    except Exception as e:
        print(f"Error during OpenAI API call: {e}")
        caption="Caption generation failed."

    chunk_filename = chunk_path.split(ANNOTATED_CHUNK_DIR+'/')[1]
    metadata_filename = chunk_filename.replace(os.path.splitext(chunk_filename)[1], '.json')
    metadata_path = os.path.join(OUTPUT_METADATA_DIR, metadata_filename)
    with open(metadata_path, 'r+') as f:
        data = json.load(f)
        data['vlm_caption'] = caption
        f.seek(0)
        json.dump(data, f, indent=4)
        f.truncate()

    embedding = encode_text([caption], processor, embedding_model, device=device)[0][0]
    pil_frames = [Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)) for frame in sampled_frames]
    image_embeddings = encode_image(pil_frames, processor, embedding_model, device=device)
    if len(image_embeddings)>1:
        avg_image_embedding = np.mean(image_embeddings, axis=0)
        avg_image_embedding = avg_image_embedding / np.linalg.norm(avg_image_embedding)
        avg_image_embedding = avg_image_embedding.tolist()
    else:
        avg_image_embedding = image_embeddings[0]

    # if embedding:
    if avg_image_embedding:
        try:
            if db_type=='chroma':
                collection.upsert(
                    # embeddings=[embedding] + [avg_image_embedding],
                    # metadatas=[
                    #     {"video_name": chunk_filename, "chunk_path": chunk_path, "caption": caption, "type":"text"},
                    #     {"video_name": chunk_filename, "chunk_path": chunk_path, "caption": caption, "type":"image"}
                    #     ],
                    # ids=[chunk_filename+"_text", chunk_filename+"_image"] # Use filename as a unique ID
                    embeddings=[avg_image_embedding],
                    metadatas=[
                        {"video_name": chunk_filename, "chunk_path": chunk_path, "caption": caption, "type":"text"},
                        ],
                    ids=[chunk_filename+"_image"] # Use filename as a unique ID
                )
            elif db_type=='milvus':
                entities = [
                    [chunk_filename+"_image"],
                    [avg_image_embedding],
                    [{"video_name": chunk_filename, "chunk_path": chunk_path, "caption": caption, "type":"image"}]
                ]
                collection.upsert(entities)
            print(f"Added embedding for {chunk_filename} to ChromaDB.")
        except Exception as e:
            print(f"Error adding to ChromaDB: {e}")
    return caption


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

async def create_aggregated_summary(video_path):
    video_name = video_path.split(DATA_DIR+'/')[1]
    print(f"Aggregating summaries for video: {video_name}")
    chunk_summaries = []

    chunk_count = 0
    metadata_files = sorted([f for f in os.listdir(OUTPUT_METADATA_DIR) if f.startswith(os.path.splitext(video_name)[0]) and f.endswith('.json')])
    for metadata_file in metadata_files:
        metadata_path = os.path.join(OUTPUT_METADATA_DIR, metadata_file)
        with open(metadata_path, 'r') as f:
            data = json.load(f)
        if data.get("motion_detected") == "True" and 'vlm_caption' in data:
            vlm_output_str = data['vlm_caption']
            vlm_output_json = parse_json(vlm_output_str)
            features = vlm_output_json.get("Features", "")
            summary = vlm_output_json.get("Video Summary", "")
            final_summary = (
                "---------------\n"
                f"Chunk {chunk_count} Features:\n"
                f"{features}\n"
                f"Chunk {chunk_count} Summary:\n"
                f"{summary}\n"
                "---------------"
            )
            if final_summary:
                chunk_summaries.append(final_summary)
            chunk_count+=1

    if not chunk_summaries:
        return
    summaries_text = "\n".join(chunk_summaries)
    aggregation_prompt = (
        "You are an expert intelligence analyst. The following are sequential summaries from a video. "
        "Synthesize them into a single, coherent paragraph describing the events of the entire video. "
        "Remove redundancies and create a logical narrative. Provide only the final summary paragraph."
        f"\n\n--- Individual Summaries ---\n{summaries_text}\n\n--- Aggregated Summary ---"
    )
    messages = [{"role": "user", "content": aggregation_prompt}]
    response = await client.chat.completions.create(
        model=LLM_MODEL_NAME, messages=messages, max_tokens=1024
    )
    final_summary = response.choices[0].message.content
    summary_path = os.path.join(OUTPUT_METADATA_DIR, f"{video_name.split('.')[0]}.json")
    with open(summary_path, 'w') as f: 
        data = {}
        data['aggregated_summary'] = final_summary
        data['video_path'] = video_path
        data['video_name'] = video_name
        f.seek(0)
        json.dump(data, f, indent=4)
        f.truncate()

    embedding = encode_text([final_summary], processor, embedding_model, device=device)[0][0]
    if embedding:
        try:
            if db_type=='chroma':
                video_collection.upsert(
                    embeddings=[embedding],
                    metadatas=[{"video_name": video_name, "summary": final_summary, "video_path": video_path}],
                    ids=[video_name] # Use filename as a unique ID
                )
            elif db_type=='milvus':
                entities = [
                    [video_name],
                    [embedding],
                    [{"video_name": video_name, "summary": final_summary, "video_path": video_path}]
                ]
                video_collection.upsert(entities)
            print(f"Added embedding for {video_name} to ChromaDB.")
        except Exception as e:
            print(f"Error adding to ChromaDB: {e}")


async def vlm_caption_all(chunk_paths):
    tasks = [vlm_caption(path) for path in chunk_paths]
    results = await asyncio.gather(*tasks)
    return results

async def llm_summarize_all(video_paths):
    tasks = [create_aggregated_summary(path) for path in video_paths]
    results = await asyncio.gather(*tasks)
    return results

# -------------------
# Pipeline for one video
# -------------------
async def process_video(video_path):
    chunks = chunk_video(video_path)
    chunks_cv = detect_and_track_objects(chunks)
    return chunks_cv

async def main():
    video_paths = []
    for filename in os.listdir(DATA_DIR):
        if filename.lower().endswith((".mp4", ".avi", ".mov")):
            video_path = os.path.join(DATA_DIR, filename)
            video_paths.append(video_path)

    tasks = [process_video(v) for v in video_paths]
    chunks_cv = await asyncio.gather(*tasks)
    chunks_cv_path = [cv[0] for cv in chunks_cv if cv is not None]
    vlm_results = await vlm_caption_all(chunks_cv_path)
    llm_results = await llm_summarize_all(video_paths)
    return None

if __name__ == "__main__":
    asyncio.run(main())