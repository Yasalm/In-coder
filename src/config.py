import os 
from pathlib import Path
import docker 


DATABASE_URI = os.environ.get("DATABASE_URI", "sqlite:///data/encoders.db")

QDRANT_HOST = os.environ.get('QDRANT_HOST', 'localhost')
QDRANT_PORT = os.environ.get('QDRANT_PORT', '6333')

folder = Path().cwd()
shared_folder = folder / 'shared'
mod_dir = shared_folder / 'onnx'
dockerfile = str(shared_folder / 'dockerfile.base')
docker_cli = docker.from_env()