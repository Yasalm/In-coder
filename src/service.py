from distutils.command.build_clib import build_clib
from qdrant_client import QdrantClient
from qdrant_client.http.models.models import Filter
from fastapi import FastAPI, Depends, Body, UploadFile, File, Form, BackgroundTasks
from src.models import Vector, Collection, Encoder, init_db, get_session, get_collection_names, get_encoder
from typing import List, Union, Optional
from enum import Enum 
from uuid import uuid4
from src.config import QDRANT_HOST, QDRANT_PORT
from sqlmodel import Session, select
from pydantic import BaseModel
from src.ops import get_types
import numpy as np
from pathlib import Path
from io import BytesIO
import docker

folder = Path().cwd()
mod_dir = folder / 'onnx'
mod_dir.mkdir(exist_ok=True)
shared_folder = folder / 'shared'
dockerfile = str(folder / 'dockerfile' / 'dockerfile')
print(dockerfile)

app = FastAPI()
client = QdrantClient(QDRANT_HOST, QDRANT_PORT)
docker_cli = docker.from_env()

class New_Collection(BaseModel):
    name: str
    desc: str
    author: str
    data: List[dict]
    encoded_fields: List[str]
    encoder_type_onnx: bool = True
    vectors : Optional[List[List[Union[int, float]]]] = None
    vector_size: Optional[int] = None
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

Collection_Names = Collection_Names("Collection_Names", {name: name for name in get_collection_names()})

def build_image(collection_name, dockerfile=dockerfile):
    tag = f'encoder:{collection_name}'.replace(' ', '_')
    _shared = str(shared_folder)
    img, buildg = docker_cli.images.build(dockerfile=dockerfile, rm=True, path=_shared, tag=tag)
    run_enocder(img, collection_name)

def run_enocder(container_image, collection_name):
    container = docker_cli.containers.run(container_image.tags[0], detach=True, name=collection_name.replace(' ', '_'), volumes={ str(shared_folder / collection_name): {'bind': '/app/onnx/', 'mode': 'ro'}})
    print(container)
    return container

def save_file(collection_name, model):
    model_loc = mod_dir / collection_name
    model_loc.mkdir(exist_ok=True)
    model_loc = model_loc / model.filename  
    try:
        with model_loc.open("wb") as f:
            while contents := model.file.read(1024*1024):
                f.write(contents)
    except Exception as e:
        print(e)
        return {"Error in uploading file",}
    finally:
        model.file.close()

@app.on_event("startup")
def on_startup():
    init_db()
    
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
    
@app.post("/collections/create",)
async def new(background_task : BackgroundTasks, collection: New_Collection = Body(...), session: Session = Depends(get_session)):
    collection_name = collection.name
    collection_author = collection.author
    collection_desc = collection.desc
    data = collection.data
    encoded_fields = collection.encoded_fields
    vector_size = collection.vector_size
    vectors = collection.vectors
    encoder_type_onnx = collection.encoder_type_onnx
    distance = collection.distance_metric

    if encoder_type_onnx:
        pass # handle onnx runtime, create collection, pass data, source_schema, 
    if not vectors:
        # content = [reco[encoded_fields[0]] for reco in data]
        # encoder = SentanceEncoder(encoded_fields=encoded_fields,)
        # vectors, vector_size = encoder.encode(content)
        pass
    background_task.add_task(build_image, collection_name)

    result = client.recreate_collection(collection_name=collection_name, vector_size=vector_size, distance=distance)
    if not isinstance(vectors, np.ndarray):
        vectors = np.array(vectors)
    _ids = [i for i in range(len(vectors))]
    client.upload_collection(collection_name=collection_name, vectors=vectors, payload=data, ids = _ids, batch_size=256, parallel=2)
    source_schema = get_types(data)
    # encoded = get_encoder(encoder_name) # for onnx its same. beacause we check file name. 
    # if encoded is None:
    #     encoded = Encoder(id=str(uuid4()), name=encoder_name, vector_size=vector_size, )

    new_collection = Collection(id=str(uuid4()), name=collection_name, author=collection_author, desc=collection_desc, encoded_fields=encoded_fields[0], source_schema=source_schema,)
    _ = [session.add(Vector(vector_id=_id, collection_id=new_collection.id)) for _id in _ids]
    session.add(new_collection)
    # session.add(encoded) 
    session.commit()
    return {
        "message": "accepted",
        "collection": collection_name,
        "id": new_collection.id
    }


@app.put("/onnx/upload",)
async def upload(background_task : BackgroundTasks,  collection_name: Collection_Names = Form(...), vector_size: int = Form(...), model: UploadFile = File(...), session: Session = Depends(get_session)):
    """
    Upload a file to the ONNX server.
    """
    collection = session.exec(select(Collection).where(Collection.name == collection_name.name)).first()
    if collection is None: return {'message': 'No collection found.'}
    background_task.add_task(save_file, collection_name.name, model)
    encoder = Encoder(id=str(uuid4()), name=model.filename, vector_size=vector_size, loc=f'{collection.name}/{model.filename}')
    collection.encoders = [encoder]
    session.add(collection)
    session.add(encoder)
    session.commit()
    return {"message": "accepted"}