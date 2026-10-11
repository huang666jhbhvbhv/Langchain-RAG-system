"""Transactional SQLite document/vector store; exact cosine search is in retrieve.py."""
from contextlib import contextmanager
import hashlib
import json
import sqlite3
import time

import numpy as np
from filelock import FileLock
from langchain_core.documents import Document

from .embedding import make_embeddings
from .ingest import scan, read_bytes, chunks


class Index:
    def __init__(self, config, embeddings=None):
        self.config = config
        config.data_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = config.data_dir / "index.sqlite3"
        self.lock = FileLock(str(config.data_dir / "index.lock"), timeout=300)
        self._embeddings = embeddings
        with self.lock, self.connect() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS files (source TEXT PRIMARY KEY, digest TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS chunks (
                    id TEXT PRIMARY KEY, source TEXT NOT NULL, content TEXT NOT NULL,
                    metadata TEXT NOT NULL, vector BLOB NOT NULL);
                CREATE INDEX IF NOT EXISTS chunk_source ON chunks(source);
            """)

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.db_path, timeout=30)
        try:
            with db:
                yield db
        finally:
            db.close()

    @property
    def embeddings(self):
        if self._embeddings is None:
            self._embeddings = make_embeddings(self.config)
        return self._embeddings

    def meta(self, db):
        return {k: json.loads(v) for k, v in db.execute("SELECT key,value FROM meta")}

    def require_config(self, meta):
        if meta.get("fingerprint") != self.config.fingerprint():
            raise ValueError("Embedding 或分块配置已变化，请用 index --rebuild 重建，或恢复原配置")

    def status(self):
        with self.connect() as db:
            return dict(**self.meta(db), files=db.execute("SELECT count(*) FROM files").fetchone()[0],
                        chunks=db.execute("SELECT count(*) FROM chunks").fetchone()[0])

    def update(self, root, rebuild=False):
        started = time.monotonic()
        with self.lock:
            root, paths = scan(root)
            with self.connect() as db:
                meta = self.meta(db)
                old = dict(db.execute("SELECT source,digest FROM files"))
            if meta:
                if meta.get("root") != str(root):
                    raise ValueError("该知识库已绑定另一个目录，请为新目录使用新的 --data-dir")
                if not rebuild:
                    self.require_config(meta)
            previous = {} if rebuild else old
            current = {}
            changed = []
            docs = []
            # Prepare everything before committing; parse/embed failure leaves old index intact.
            for path in paths:
                source = path.relative_to(root).as_posix()
                data = read_bytes(path)
                digest = hashlib.sha256(data).hexdigest()
                current[source] = digest
                if previous.get(source) != digest:
                    changed.append(source)
                    docs.extend(chunks(data, source, self.config))
                    if len(docs) > 20000:
                        raise ValueError("本次更新超过 20000 个片段，请按项目拆分知识库")
            removed = sorted(set(old) - set(current))
            if not changed and not removed and meta and not rebuild:
                return dict(changed=0, removed=0, unchanged=len(current), embedded_chunks=0,
                            seconds=round(time.monotonic()-started, 3))
            vectors = []
            dim = None if rebuild else meta.get("dimension")
            for start in range(0, len(docs), 32):
                batch = docs[start:start+32]
                values = self.embeddings.embed_documents([d.page_content for d in batch])
                if len(values) != len(batch):
                    raise ValueError("Embedding 返回数量与输入不一致")
                for value in values:
                    v = np.asarray(value, dtype=np.float32)
                    if v.ndim != 1 or not len(v) or not np.all(np.isfinite(v)):
                        raise ValueError("Embedding 返回无效向量")
                    dim = dim or len(v)
                    if len(v) != dim:
                        raise ValueError("向量维度不一致，需要重建索引")
                    norm = np.linalg.norm(v)
                    if norm == 0:
                        raise ValueError("Embedding 返回零向量")
                    vectors.append((v/norm).astype(np.float32).tobytes())
            new_meta = dict(root=str(root), fingerprint=self.config.fingerprint(), dimension=dim,
                            updated_at=time.time(), revision=meta.get("revision", 0)+1)
            with self.connect() as db:
                if rebuild:
                    db.execute("DELETE FROM chunks")
                    db.execute("DELETE FROM files")
                for source in set(changed+removed):
                    db.execute("DELETE FROM chunks WHERE source=?", (source,))
                    db.execute("DELETE FROM files WHERE source=?", (source,))
                db.executemany("INSERT INTO files VALUES (?,?)", [(s, current[s]) for s in changed])
                db.executemany("INSERT INTO chunks VALUES (?,?,?,?,?)", [
                    (d.metadata["id"], d.metadata["source"], d.page_content,
                     json.dumps(d.metadata, ensure_ascii=False), v) for d, v in zip(docs, vectors)])
                db.executemany("INSERT OR REPLACE INTO meta VALUES (?,?)",
                    [(k, json.dumps(v, ensure_ascii=False)) for k, v in new_meta.items()])
            return dict(changed=len(changed), removed=len(removed), unchanged=len(current)-len(changed),
                        embedded_chunks=len(docs), seconds=round(time.monotonic()-started, 3))

    def snapshot(self):
        with self.connect() as db:
            db.execute("BEGIN")  # Keep metadata and vectors from one committed version.
            meta = self.meta(db)
            if not meta:
                raise ValueError("尚未建立索引，请先导入文档")
            self.require_config(meta)
            rows = db.execute("SELECT content,metadata,vector FROM chunks ORDER BY id").fetchall()
        docs = [Document(page_content=c, metadata=json.loads(m)) for c, m, _ in rows]
        matrix = np.vstack([np.frombuffer(v, dtype=np.float32) for _, _, v in rows]) if rows else np.empty((0,0))
        return meta, docs, matrix
