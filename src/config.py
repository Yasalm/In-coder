import os 

DATABASE_URI = os.environ.get("DATABASE_URI", "sqlite:///incodes.db")

QDRANT_HOST = os.environ.get('QDRANT_HOST', 'localhost')
QDRANT_PORT = os.environ.get('QDRANT_PORT', '6333')