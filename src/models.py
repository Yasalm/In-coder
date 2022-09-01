from typing import Optional, List
from sqlmodel import Field, SQLModel, Column, create_engine, JSON, Session, Relationship, select
from src.config import DATABASE_URI
import time 

class EncoderVersion:
    def __init__(self, version: str = "v0" ):
        self.version = version
    def upgrade(self,):
        curr_version = self.version[1:]
        new_version = int(curr_version)
        new_version += 1
        self.version = f'v{new_version}'
    def downgrade(self,):
        curr_version = self.version[1:]
        new_version = int(curr_version)
        new_version -= 1
        self.version = f'v{new_version}'
    def __repr__(self) -> str:
        return str(self.version)

class EncodingLink(SQLModel, table=True):
    colleciton_id : Optional[str] = Field(default=None, foreign_key='collection.id', primary_key=True)
    encoder_id : Optional[str] = Field(default=None, foreign_key='encoder.id', primary_key=True)

class Encoder(SQLModel, table=True):
    id: Optional[str] = Field(default=None, primary_key=True)
    name : str = Field(nullable=False)
    loc : str = Field(nullable=False)
    is_onnx: bool = Field(default=True)
    vector_size : int = Field(nullable=False)
    created_at : float = Field(default=time.time(), )

    collections : List["Collection"] = Relationship(back_populates="encoders", link_model=EncodingLink)

class Collection(SQLModel, table=True):
    id: Optional[str] = Field(default=None, primary_key=True)
    name : str = Field(nullable=False, index=True)
    author : str = Field(nullable=False, index=True)
    version: str = Field(default=str(EncoderVersion()), index=True)
    desc : str = Field(nullable=False, )
    encoded_fields: str = Field(nullable=False, )
    source_schema: List[dict] = Field(default={}, sa_column=Column(JSON),)
    created_at : float = Field(default=time.time(), )
    is_active : Optional[bool] = Field(default=True,)

    encoders : List[Encoder] = Relationship(back_populates="collections", link_model=EncodingLink)

    class Config:
        arbitrary_types_allowed = True

class Vector(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    collection_id: str = Field(foreign_key="collection.id")
    vector_id : int = Field(index=True)

class CallHistory(SQLModel, table=True):
    id: Optional[int] = Field(primary_key=True)
    collection_id: str = Field(foreign_key="collection.id")
    vector_id: int = Field(foreign_key="vector.id")
    search_field : str = Field(nullable=False)
    created_at : float = Field(default=time.time(), )



engine = create_engine(DATABASE_URI, connect_args={"check_same_thread": False})
def init_db():
    SQLModel.metadata.create_all(engine)

def get_session():
    with Session(engine) as session:
        yield session

def get_collection_names():
    session = next(get_session())
    return session.exec(select(Collection.name)).all()

def get_encoder(encoder_name):
    session = next(get_session())
    return session.exec(select(Encoder).where(Encoder.name == encoder_name)).first()