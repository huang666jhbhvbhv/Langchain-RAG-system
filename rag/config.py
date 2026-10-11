from dataclasses import dataclass, field
from pathlib import Path
import os

from dotenv import load_dotenv

APP = Path(__file__).resolve().parents[1]


@dataclass
class Config:
    data_dir: Path = APP / ".rag-data" / "default"
    provider: str = "local"
    embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    embedding_key: str = field(default="", repr=False)
    embedding_url: str = ""
    llm_key: str = field(default="", repr=False)
    llm_url: str = "https://api.deepseek.com"
    llm_model: str = "deepseek-flash"
    timeout: int = 90
    max_tokens: int = 2048
    chunk_size: int = 800
    overlap: int = 120
    top_k: int = 5
    context_chars: int = 10000
    min_cosine: float = 0.25
    min_bm25: float = 0.1
    demo: bool = False

    def __post_init__(self):
        self.data_dir = Path(self.data_dir).resolve()
        if not 0 <= self.overlap < self.chunk_size or self.chunk_size < 100:
            raise ValueError("必须满足 chunk_size >= 100 且 0 <= overlap < chunk_size")
        if not 1 <= self.top_k <= 20 or self.context_chars < self.chunk_size:
            raise ValueError("top_k 必须为 1..20，context_chars 不能小于 chunk_size")

    @classmethod
    def load(cls, data_dir=None, demo=False):
        load_dotenv(APP / ".env", override=False)
        return cls(
            data_dir=Path(data_dir) if data_dir else APP / ".rag-data" / ("demo" if demo else "default"),
            provider="demo" if demo else os.getenv("EMBEDDING_PROVIDER", "local"),
            embedding_model="hash-demo-v1" if demo else os.getenv("EMBEDDING_MODEL", cls.embedding_model),
            embedding_key=os.getenv("EMBEDDING_API_KEY", ""),
            embedding_url=os.getenv("EMBEDDING_BASE_URL", ""),
            llm_key=os.getenv("LLM_API_KEY", ""), llm_url=os.getenv("LLM_BASE_URL", cls.llm_url),
            llm_model=os.getenv("LLM_MODEL", cls.llm_model), timeout=int(os.getenv("LLM_TIMEOUT", "90")),
            max_tokens=int(os.getenv("LLM_MAX_TOKENS", "2048")),
            chunk_size=int(os.getenv("CHUNK_SIZE", "800")), overlap=int(os.getenv("CHUNK_OVERLAP", "120")),
            top_k=int(os.getenv("TOP_K", "5")), context_chars=int(os.getenv("CONTEXT_CHARS", "10000")),
            min_cosine=float(os.getenv("MIN_COSINE", "0.25")), min_bm25=float(os.getenv("MIN_BM25", "0.1")), demo=demo)

    def fingerprint(self):
        # No keys: changing credentials should not invalidate compatible vectors.
        return dict(schema=1, provider=self.provider, model=self.embedding_model,
                    endpoint=self.embedding_url if self.provider == "openai" else "",
                    chunk_size=self.chunk_size, overlap=self.overlap)
