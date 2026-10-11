import json
from pathlib import Path
import time

from .answer import RAG
from .retrieve import search


def evaluate(index, questions_file, modes=("vector", "keyword", "hybrid"), answers=False):
    questions = [json.loads(line) for line in Path(questions_file).read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    if not questions:
        raise ValueError("评测问题集为空")
    for q in questions:
        if not isinstance(q.get("question"), str) or not isinstance(q.get("sources"), list):
            raise ValueError("每条问题需要 question 和 sources；无答案问题 sources=[]")
    report = dict(demo=index.config.demo, top_k=index.config.top_k, fingerprint=index.config.fingerprint(),
                  definition="source-level retrieval metrics; keyword coverage is NOT answer correctness", modes={})
    rag = RAG(index)
    for mode in modes:
        rows = []
        for q in questions:
            started = time.monotonic()
            result = rag.ask(q["question"], mode) if answers else search(index, q["question"], mode)
            hits = result["hits"]
            expected = set(q["sources"])
            retrieved = [h["source"] for h in hits]
            ranks = [i for i, source in enumerate(retrieved, 1) if source in expected]
            row = dict(id=q.get("id"), question=q["question"], expected=sorted(expected), retrieved=retrieved,
                       seconds=round(time.monotonic()-started, 3),
                       source_recall=len(expected & set(retrieved))/len(expected) if expected else None,
                       hit=bool(ranks) if expected else None, rr=1/min(ranks) if ranks else 0,
                       empty=not hits)
            if answers:
                terms = q.get("answer_terms", [])
                row.update(answer=result["answer"], status=result["status"], citations=result["citations"],
                    usage=result["usage"], term_coverage=sum(t.casefold() in result["answer"].casefold() for t in terms)/len(terms) if terms else None)
            rows.append(row)
        known = [r for r in rows if r["expected"]]
        unknown = [r for r in rows if not r["expected"]]
        report["modes"][mode] = dict(
            questions=len(rows), answerable=len(known), unanswerable=len(unknown),
            hit_at_k=sum(r["hit"] for r in known)/len(known) if known else None,
            source_recall_at_k=sum(r["source_recall"] for r in known)/len(known) if known else None,
            mrr_at_k=sum(r["rr"] for r in known)/len(known) if known else None,
            unanswerable_empty_rate=sum(r["empty"] for r in unknown)/len(unknown) if unknown else None,
            mean_seconds=sum(r["seconds"] for r in rows)/len(rows), rows=rows)
    return report
