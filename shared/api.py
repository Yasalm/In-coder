from loguru import logger
from fastapi import FastAPI
import uvicorn
from config import COLLECTION_NAME, QDRANT_HOST, QDRANT_PORT, MODEL_NAME, MODEL_PATH, MODEL_TYPE
from qdrant_client import QdrantClient
from typing import List, Optional
from pydantic import BaseModel
from fastapi import FastAPI, Body
from qdrant_client.http.models.models import Filter
import numpy as np
from encoder import sentence_transformers_onnx, Encoder
from sentence_transformers import SentenceTransformer
import onnxruntime 

client = QdrantClient(QDRANT_HOST, QDRANT_PORT)
app = FastAPI()
app.encoder = None



@app.on_event("startup")
def on_startup():
    if MODEL_TYPE == 'SENTANCE_TRANSFORMER':
        app.encoder = sentence_transformers_onnx(
            model = SentenceTransformer(MODEL_NAME),
            path = str(MODEL_PATH),
        )
    elif MODEL_TYPE == 'ONNX':
        model_path = str(MODEL_PATH / MODEL_NAME)
        app.encoder = Encoder(onnxruntime.InferenceSession(model_path))

class Query(BaseModel):
    content : List
    with_metadata : bool = True
    with_vector : bool = False
    top : int = 5
    filters : Optional[dict] = None
    encode : bool = True

@app.post("/query",)
async def query(query: Query = Body(),):
    vectors = query.content
    filters = query.filters
    topk = query.top
    with_vector = query.with_vector
    with_metadata = query.with_metadata

    if not isinstance(vectors[0], str):
        vectors = np.array(vectors)

    vectors = app.encoder.encode(vectors,)

    client_result = client.search(collection_name=COLLECTION_NAME, query_vector=vectors, query_filter=Filter(**filters) 
    if filters else None, top=topk, with_vector=with_vector, with_payload=with_metadata)
    return client_result

@app.get('/')
def status():
    message = {"status": "OK","service": f"{COLLECTION_NAME}-{MODEL_NAME}" }
    return message


if __name__ == "__main__":
    logger.info(f"Starting to serve requests for collection {COLLECTION_NAME}")
    uvicorn.run(app, host="0.0.0.0", port=8080)