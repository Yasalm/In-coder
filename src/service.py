from qdrant_client import QdrantClient
from qdrant_client.http.models.models import Filter
from fastapi import FastAPI, Depends, Body, UploadFile, File, Form, BackgroundTasks
from models import Vector, Collection, Encoder, init_db, get_session, get_collection_names, is_collection, has_encoder, get_encoders
from typing import List, Union, Optional
from enum import Enum 
from uuid import uuid4
from config import QDRANT_HOST, QDRANT_PORT, mod_dir
from container import *
from sqlmodel import Session, select
from pydantic import BaseModel
from types import get_types
import numpy as np
import uvicorn
from loguru import logger
import sys

app = FastAPI()
client = QdrantClient(QDRANT_HOST, QDRANT_PORT)


class New_Collection(BaseModel):
    name: str
    desc: str
    author: str
    data: List[dict]
    encoded_fields: List[str]
    transformer_model_name: Optional[str] = None
    vectors : Optional[List[List[Union[int, float]]]]
    vector_size: Optional[int]
    distance_metric : Optional[str] = "Cosine" 

class Collection_Result(BaseModel):
    # contains the result of metadata + client result
    pass

class Collection_Created(BaseModel):
    status: str

class Query(BaseModel):
    content : Union[str, List[str]]
    vectors: List[List[Union[int, float]]]
    with_metadata : bool = True
    with_vector : bool = False
    top : int = 5
    filters : Optional[dict] = None
    encode : bool = True

class Collection_Names(Enum):
    ...

try:
    Collection_Names =  Collection_Names("Collection_Names", {name: name for name in get_collection_names()})
except:
    pass


def save_file(collection_name, model):
    model_loc = mod_dir / collection_name
    model_loc.mkdir(exist_ok=True)
    model_loc = model_loc / model.filename
    try:
        with model_loc.open("wb") as f:
            while contents := model.file.read(1024*1024):
                f.write(contents)
    except Exception as e:
        logger.error(f"Error uploading file: {e}")
        return {"Error in uploading file",}
    finally:
        model.file.close()

@app.on_event("startup")
def on_startup():
    logger.info("Setting up database")
    init_db()
    logger.info("Setting up base container image..")
    setup_base_image()
    
@app.get('/')
def status(session : Session = Depends(get_session)):
    message = {"status": "OK"}
    collections = client.get_collections().collections
    collections_info = []
    for collection in collections:
        collection = session.exec(select(Collection).where(Collection.name == collection.name)).first()
        collections_info.append({'collection': collection.name, "desc": collection.desc, 'author': collection.author, 'version': collection.version,'encoded_fields': collection.encoded_fields, 'source_schema': collection.source_schema,  'encoder': collection.encoders[0].name,})
    message.update({"collections": collections_info})
    return message

@app.get("/collections/",)
async def get_collections():
    colletcions = client.get_collections()
    return colletcions

@app.get("/collections/{collection_name}", response_model=Collection)
async def get_collection(collection_name: Collection_Names, session: Session = Depends(get_session)):
    collection = session.exec(select(Collection).where(Collection.name == collection_name.name)).first()
    if collection is None:
        return {
            "Message": "There are no collections of This name."
        }
    return collection

@app.post("/collections/{collection_name}/query",  )
async def query(collection_name : Collection_Names, query : Query = Body(), session: Session = Depends(get_session)):
    collection_name = collection_name.name
    content = query.content
    vectors = query.vectors
    filters = query.filters
    topk = query.top
    with_vector = query.with_vector
    with_metadata = query.with_metadata
    encode = query.encode

    if encode and content is not None:
        encoder_name = session.exec(select(Collection).where(Collection.name == collection_name)).first().encoders[0].name
        if not encoder_name:
            return {
                "status": "Error",
                "message": f"Can not find collection named {collection_name}"
            }

    client_result = client.search(collection_name=collection_name, query_vector=vectors, query_filter=Filter(**filters) 
    if filters else None, top=topk, with_vector=with_vector, with_payload=with_metadata)
    return client_result
    
@app.post("/collections/new",)
async def new(background_task : BackgroundTasks, collection: New_Collection = Body(...), session: Session = Depends(get_session)):
    collection_name = collection.name
    collection_author = collection.author
    collection_desc = collection.desc
    data = collection.data
    encoded_fields = collection.encoded_fields
    vector_size = collection.vector_size
    vectors = collection.vectors
    distance = collection.distance_metric
    transformer_model_name = collection.transformer_model_name

    collection_name = collection_name.replace(' ', '_')

    if is_collection(collection_name=collection_name):
        return {
            "message": "not accepted, collection already exists",
            "collection": "",
            "id": ""
        }
    if not isinstance(vectors, np.ndarray):
        vectors = np.array(vectors)

    logger.info(f"creating new collection {collection_name}, with vector size {vector_size}")
    if vector_size is None:
        vector_size = 768

    response = client.recreate_collection(collection_name=collection_name, vector_size=vector_size, distance=distance)

    _ids = [i for i in range(len(vectors))]

    client.upload_collection(collection_name=collection_name, vectors=vectors, payload=data, ids = _ids, batch_size=256, parallel=2)
    logger.info(f"uploaded data and vectors to collection {collection_name}")

    encoder = None
    if transformer_model_name:
        encoder = Encoder(id=str(uuid4()), name=transformer_model_name, vector_size=vector_size, is_onnx=False)
        logger.info(f"starting a container {collection_name}-{transformer_model_name}")
        background_task.add_task(run_contaier, collection_name=collection_name, model_name=transformer_model_name, model_type="SENTANCE_TRANSFORMER")

    source_schema = get_types(data)

    new_collection = Collection(id=str(uuid4()), name=collection_name, author=collection_author, desc=collection_desc, encoded_fields=encoded_fields[0], source_schema=source_schema,)
    if encoder:
        new_collection.encoders = [encoder]

    _ = [session.add(Vector(vector_id=_id, collection_id=new_collection.id)) for _id in _ids]

    session.add(new_collection)
    session.commit()

    logger.info("added collection metadata to database")

    return {
        "message": "accepted",
        "collection": collection_name,
        "id": new_collection.id
    }


@app.get("/collections/{collection_name}/{model_name}/dectivate",)
async def deactivate(collection_name: str, model_name: str, session: Session = Depends(get_session)):
    collection_name = collection_name.replace(' ', '_')
    if not is_collection(collection_name=collection_name):
        return {
            "message": "not accepted, collection not created",
            "collection": "",
            "id": ""
        }
    collection = session.exec(select(Collection).where(Collection.name == collection_name)).first()
    model_name  = collection.encoders[0].name
    name = f'{collection_name}-{model_name}'
    container, status = get_container(name_or_id=name)
    if container and status in ['paused', 'exited']:
        return {
            "message": "container not running, status: %s" % status,
            "collection": collection_name,
            "id": container.id
        }
    logger.info("stopping container %s", name)
    container.stop()
    container.reload()
    return {
            "message": "container stopped successfully",
            "collection": collection_name,
            "id": container.id
        }

@app.get("/collections/{collection_name}/{model_name}/activate",)
async def activate(collection_name: str, model_name: str, session: Session = Depends(get_session)):
    collection_name = collection_name.replace(' ', '_')
    if not is_collection(collection_name=collection_name):
        return {
            "message": "not accepted, collection not created",
            "collection": "",
            "id": ""
        }
    collection = session.exec(select(Collection).where(Collection.name == collection_name)).first()
    model_name  = collection.encoders[0].name
    name = f'{collection_name}-{model_name}'
    container, status = get_container(name_or_id=name)
    if container and status == 'running':
        return {
            "message": "container already running",
            "collection": collection_name,
            "id": container.id
        }
    logger.info("starting container %s", name)
    container.start()
    container.reload()
    return {
            "message": "container started successfully",
            "collection": collection_name,
            "id": container.id
        }


@app.put("/onnx/upload",)
async def upload(background_task : BackgroundTasks,  collection_name: str = Form(...), vector_size: int = Form(...), model: UploadFile = File(...), session: Session = Depends(get_session)):
    """
    Upload a file to the ONNX server.
    """
    
    collection_name = collection_name.replace(' ', '_')
    if not is_collection(collection_name=collection_name):
        return {
            "message": "not accepted, collection not created",
            "collection": "",
            "id": ""
        }
    if has_encoder(encoder_name=model.filename):
        return {
            "message": f"not accepted, {model.filename} already uploaded to {collection_name}",
            "collection": "",
            "id": ""
        }
    collection = session.exec(select(Collection).where(Collection.name == collection_name)).first()
    logger.info(f"saving an onnx file in a background task for {collection_name}")
    background_task.add_task(save_file, collection_name, model)
    encoder = Encoder(id=str(uuid4()), name=model.filename, vector_size=vector_size, loc=f'{collection_name}/{model.filename}', collections=[collection])
    session.add(collection)
    session.add(encoder)
    session.commit()
    logger.info(f"added onnx {model.filename} to {collection_name}")
    model_name = model.filename.split('.')[0]
    logger.info(f"starting a container {collection_name}-{model_name}")
    background_task.add_task(run_contaier, collection_name=collection_name, model_name=model.filename, model_type="ONNX")
    return {
            "message": "accepted",
            "collection": collection_name,
            "onnx_name": model.filename,
            "onnx_id": encoder.id
            }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level='info')