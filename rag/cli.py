import argparse
import json
from pathlib import Path
import sys

from .answer import RAG
from .config import APP, Config
from .evaluate import evaluate
from .retrieve import search
from .store import Index


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="项目文档 RAG；全局参数放在子命令之前")
    parser.add_argument("--demo", action="store_true", help="离线词法向量/原文展示，绝不调用真实 LLM")
    parser.add_argument("--data-dir", help="独立知识库索引目录")
    sub = parser.add_subparsers(dest="command", required=True)
    idx = sub.add_parser("index", help="增量同步文档目录，删除源文件会删除对应索引")
    idx.add_argument("folder", type=Path)
    idx.add_argument("--rebuild", action="store_true", help="重新计算全部向量；成功前保留旧索引")
    for name in ("search", "ask"):
        cmd = sub.add_parser(name)
        cmd.add_argument("question")
        cmd.add_argument("--mode", choices=("hybrid", "vector", "keyword"), default="hybrid")
    sub.add_parser("status")
    sub.add_parser("doctor")
    ev = sub.add_parser("evaluate")
    ev.add_argument("--questions", type=Path, default=APP/"evaluation/questions.jsonl")
    ev.add_argument("--answers", action="store_true", help="同时生成答案：正式模式消耗真实 API 额度")
    ev.add_argument("--k", type=int, default=3, choices=range(1,21), help="评测检索数量，默认 3")
    ev.add_argument("--output", type=Path, default=APP/"reports/evaluation.json")
    args = parser.parse_args()
    try:
        config = Config.load(args.data_dir, args.demo)
        if args.command == "doctor":
            import importlib.util
            value = dict(python=sys.version.split()[0], app=str(APP), data_dir=str(config.data_dir),
                provider=config.provider, embedding_model=config.embedding_model, llm_model=config.llm_model,
                llm_key_configured=bool(config.llm_key and not config.llm_key.startswith("YOUR_")),
                local_embedding_installed=importlib.util.find_spec("langchain_huggingface") is not None,
                note="仅检查配置，不发送网络请求，不加载模型")
        else:
            index = Index(config)
            if args.command == "index":
                value = index.update(args.folder, args.rebuild)
            elif args.command == "status":
                value = index.status()
            elif args.command == "search":
                value = search(index, args.question, args.mode)
            elif args.command == "ask":
                value = RAG(index).ask(args.question, args.mode)
            else:
                config.top_k = args.k
                value = evaluate(index, args.questions, answers=args.answers)
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
                value = dict(output=str(args.output.resolve()), demo=config.demo,
                    modes={m: {k:v for k,v in data.items() if k != "rows"} for m,data in value["modes"].items()})
        print(json.dumps(value, ensure_ascii=False, indent=2))
    except Exception as exc:
        # Provider errors can contain requests. Redact configured keys before display.
        text = str(exc)
        if "config" in locals():
            for key in (config.llm_key, config.embedding_key):
                if key:
                    text = text.replace(key, "[REDACTED]")
        print(f"Error: {text}", file=sys.stderr)
        return 1
    return 0
