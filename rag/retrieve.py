import numpy as np
from rank_bm25 import BM25Plus

from .embedding import tokenize


def search(index, query, mode="hybrid", k=None):
    query = query.strip()
    if not query or len(query) > 8000:
        raise ValueError("问题长度必须为 1..8000 字符")
    if mode not in ("hybrid", "vector", "keyword"):
        raise ValueError("检索方式为 hybrid/vector/keyword")
    k = index.config.top_k if k is None else int(k)
    if not 1 <= k <= 20:
        raise ValueError("k 必须为 1..20")
    meta, docs, matrix = index.snapshot()
    if not docs:
        return dict(query=query, hits=[], revision=meta["revision"], mode=mode)
    n = len(docs)
    cos = np.zeros(n)
    lexical = np.zeros(n)
    orders = []
    if mode in ("hybrid", "vector"):
        v = np.asarray(index.embeddings.embed_query(query), dtype=np.float32)
        if v.ndim != 1 or v.shape[0] != matrix.shape[1] or not np.all(np.isfinite(v)):
            raise ValueError("查询向量维度或数值错误，请检查 Embedding 配置")
        norm = np.linalg.norm(v)
        if norm:
            cos = matrix @ (v/norm)
        orders.append(sorted(range(n), key=lambda i: (-cos[i], i))[:max(k*4, 20)])
    if mode in ("hybrid", "keyword"):
        corpus = [tokenize(d.page_content) or ["__empty__"] for d in docs]
        words = tokenize(query)
        if words:
            lexical = BM25Plus(corpus, delta=0).get_scores(words)
            orders.append(sorted(range(n), key=lambda i: (-lexical[i], i))[:max(k*4, 20)])
    scores = {}
    for order in orders:
        for rank, i in enumerate(order, 1):
            scores[i] = scores.get(i, 0) + 1/(60+rank)
    # Use raw channel scores for gating; RRF scores are ranks, not confidence.
    eligible = [i for i in scores if
        (mode != "keyword" and cos[i] >= index.config.min_cosine) or
        (mode != "vector" and lexical[i] >= index.config.min_bm25)]
    ranked = sorted(eligible, key=lambda i: (-scores[i], i))[:k]
    hits = [dict(text=docs[i].page_content, **docs[i].metadata,
                 cosine=round(float(cos[i]), 5), bm25=round(float(lexical[i]), 5),
                 rrf=round(scores[i], 6)) for i in ranked]
    return dict(query=query, hits=hits, revision=meta["revision"], mode=mode)
