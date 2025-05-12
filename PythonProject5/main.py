import os
import cv2
import uuid
import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, Filter, FieldCondition, MatchValue

from qdrant_setup import create_collection

app = FastAPI()
client = QdrantClient("http://localhost:6333")
collection_name = "video_frames"
output_dir = "output_frames"
os.makedirs(output_dir, exist_ok=True)

create_collection()


def extract_frames(video_path: str, interval: int = 1):
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_count = 0
    saved_paths = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % (fps * interval) == 0:
            filename = f"{uuid.uuid4()}.jpg"
            filepath = os.path.join(output_dir, filename)
            cv2.imwrite(filepath, frame)
            saved_paths.append(filepath)
        frame_count += 1
    cap.release()
    return saved_paths


def compute_feature_vector(image_path: str):
    image = cv2.imread(image_path)
    hist = cv2.calcHist([image], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    hist = cv2.normalize(hist, hist).flatten()
    return hist.astype(np.float32)


@app.post("/upload_video/")
async def upload_video(file: UploadFile = File(...)):
    if not file.filename.endswith(".mp4"):
        raise HTTPException(status_code=400, detail="Only MP4 files are supported.")

    video_path = os.path.join(output_dir, file.filename)
    with open(video_path, "wb") as f:
        f.write(await file.read())

    frame_paths = extract_frames(video_path)
    points = []

    for frame_path in frame_paths:
        vector = compute_feature_vector(frame_path)
        points.append(PointStruct(
            id=str(uuid.uuid4()),
            vector=vector.tolist(),
            payload={"image_path": frame_path}
        ))

    client.upsert(collection_name=collection_name, points=points)
    return {"message": f"{len(points)} frames extracted and indexed."}


@app.post("/search_similar/")
async def search_similar(file: UploadFile = File(...), top_k: int = 5):
    temp_path = os.path.join(output_dir, "temp.jpg")
    with open(temp_path, "wb") as f:
        f.write(await file.read())

    vector = compute_feature_vector(temp_path)
    search_result = client.search(
        collection_name=collection_name,
        query_vector=vector.tolist(),
        limit=top_k
    )

    results = []
    for point in search_result:
        image_path = point.payload['image_path']
        results.append({
            "score": point.score,
            "image_path": image_path,
            "feature_vector": point.vector
        })

    return results


@app.get("/get_frame/")
async def get_frame(path: str):
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Image not found.")
    return FileResponse(path, media_type="image/jpeg")
