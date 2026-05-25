import os
from FlagEmbedding import BGEM3FlagModel
import torch

_DEFAULT_DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
_ENV_DEVICE = os.environ.get("BGE_DEVICE", "").strip().lower()
if _ENV_DEVICE:
    _DEFAULT_DEVICE = _ENV_DEVICE


class BGEM3Encoder:
    def __init__(self, model_name="BAAI/bge-m3", use_fp16=True, devices=None):
        if devices is None:
            devices = _DEFAULT_DEVICE
        # fp16 only safe on CUDA; force fp32 on CPU.
        if isinstance(devices, str) and devices.lower().startswith("cpu"):
            use_fp16 = False
        self.devices = devices
        self.model = BGEM3FlagModel(model_name, use_fp16=use_fp16, devices=devices)
        self.model_name = model_name
        print(f"[BGEM3Encoder] model={model_name} | device={devices} | fp16={use_fp16}")

    def encode_chunks(self, chunks, batch_size = 32, max_length = 8192):
        texts = []
        for chunk in chunks:
            text = str(chunk["content_text"]) # encode chỉ xử lý dạng text
            texts.append(text)
        
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            max_length=max_length,
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=False,
        )

        # Thêm embedding vào chunks
        for i in range(len(chunks)):
            # Dense: np array → list
            chunks[i]["dense_embedding"] = embeddings["dense_vecs"][i].tolist()

            # Sparse: {int: float} format
            sparse = embeddings["lexical_weights"][i]
            sparse_dict = {}
            for k, v in sparse.items():
                sparse_dict[int(k)] = float(v)
            chunks[i]["sparse_embedding"] = sparse_dict

        return chunks