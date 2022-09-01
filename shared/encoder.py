import numpy as np
from typing import Union, List, Dict, Any, Tuple
import abc
import importlib
import onnxruntime
from pathlib import Path

# to facilitate further types of encoders and provider a general interface, beyond sentence_transformers 
class EncoderABC(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def encode(self, content: Union[List[str], str], batch_size : int = 64,):
        pass
    @abc.abstractmethod
    def load_model():
        pass

class OnnxEncoder(EncoderABC):
    def __init__(self, collection_name, model_path: str, ):
        self.model_path = model_path
        self.collection_name = collection_name
        self.session = self.load_model(model_path)
        self.input_name = self.session.get_inputs()[0].name

    def encode(self, content : Union[List[str], str], batch_size : int):
        return self.session.run(None, {self.input_name: content})

    def load_model(self, model_path : str):
        mod_dir = Path().cwd() / 'onnx'
        return onnxruntime.InferenceSession(mod_dir / model_path)
        

class SentanceEncoder(EncoderABC):
    def __init__(self, encoded_fields : List[str] = None, device: str = 'cpu', model_name: str = None):
        self.encoded_fields = encoded_fields
        self.device = device
        self.sentence_transformers = importlib.import_module("sentence_transformers")
        self.load_model(model_name)
    def load_model(self, model_name: str):
        self.model =  self.sentence_transformers.SentenceTransformer(model_name, device=self.device)
        return self.model
    def encode(self, content : Union[List[str], str], batch_size : int):
        if isinstance(content, list) and len(content) > 1:
            vectors = []
            batch = []
            for i_content in content:
                batch.append(i_content)
                if len(batch) > batch_size:
                    vectors.append(self.model.encode(batch))
                    batch = []
            if len(batch) > 0:
                vectors.append(self.model.encode(batch))
                batch = []
            vectors = np.concatenate(vectors)
            return vectors, vectors.shape[1]
        vectors = self.model.encode(content).tolist()
        return vectors, len(vectors)