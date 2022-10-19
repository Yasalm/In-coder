import os 
from pathlib import Path

QDRANT_HOST = os.environ.get('QDRANT_HOST', 'host.docker.internal')
QDRANT_PORT = os.environ.get('QDRANT_PORT', '6333')

COLLECTION_NAME = os.environ.get('COLLECTION_NAME',)
MODEL_NAME = os.environ.get('MODEL_NAME',)
MODEL_TYPE = os.environ.get('MODEL_TYPE',)

MODEL_PATH = Path().cwd()  / 'onnx' 