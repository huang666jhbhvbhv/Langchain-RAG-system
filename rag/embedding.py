import hashlib
import re

import numpy as np
from langchain_core.embeddings import Embeddings


def tokenize(text):
    """English identifiers + Chinese unigrams/bigrams; deliberately no external segmenter."""
    tokens = re.findall(r"[a-z0-9_]+", text.lower())
    for run in re.findall(r"[\u4e00-\u9fff]+", text):
        tokens.extend(run)
        tokens.extend(run[i:i+2] for i in range(len(run)-1))
    return tokens


class DemoEmbeddings(Embeddings):
    """Deterministic lexical hashing for OFFLINE PIPELINE TESTS, not semantic embeddings."""
    def embed_documents(self, texts):
        result = []
        for text in texts:
            v = np.zeros(512, dtype=np.float32)
            for word in tokenize(text):
                pos = int.from_bytes(hashlib.sha256(word.encode()).digest()[:4], "big") % len(v)
                v[pos] += 1
            norm = np.linalg.norm(v)
            result.append((v / norm if norm else v).tolist())
        return result

    def embed_query(self, text):
        return self.embed_documents([text])[0]


def make_embeddings(config):
    if config.provider == "demo":
        if not config.demo:
            raise ValueError("Demo Embeddings 只能通过显式 --demo / 演示模式启用")
        return DemoEmbeddings()
    if config.provider == "local":
        try:
            from langchain_huggingface import HuggingFaceEmbeddings
        except ImportError:
            raise RuntimeError("本地向量模型未安装：python -m pip install -r requirements-local.txt") from None
        return HuggingFaceEmbeddings(model_name=config.embedding_model,
            model_kwargs={"device": "cpu", "trust_remote_code": False},
            encode_kwargs={"normalize_embeddings": True}, show_progress=False)
    if config.provider == "openai":
        if not config.embedding_key or not config.embedding_url:
            raise ValueError("请单独配置 EMBEDDING_API_KEY、EMBEDDING_BASE_URL 和 EMBEDDING_MODEL")
        from langchain_openai import OpenAIEmbeddings
        return OpenAIEmbeddings(model=config.embedding_model, api_key=config.embedding_key,
            base_url=config.embedding_url, check_embedding_ctx_length=False,
            request_timeout=config.timeout, max_retries=2)
    raise ValueError("EMBEDDING_PROVIDER 必须为 local 或 openai；离线使用 --demo")
