"""Local Streamlit UI. Launch: python -m streamlit run app.py"""
import hashlib
import json
from pathlib import Path
import re

import streamlit as st

from rag.answer import RAG
from rag.config import APP, Config
from rag.ingest import EXTENSIONS, MAX_BYTES
from rag.retrieve import search
from rag.store import Index


def show_sources(sources):
    for number, source in enumerate(sources, 1):
        pos = f"第 {source['page']} 页" if source["page"] else f"第 {source['line_start']}–{source['line_end']} 行"
        with st.expander(f"[{source.get('citation', number)}] {source['source']} · {pos}"):
            st.code(source["text"], language=None)
            st.caption(f"cosine={source['cosine']} · BM25={source['bm25']} · RRF={source['rrf']}")


def upload_files(folder, uploads, overwrite=False):
    folder = Path(folder).resolve()
    folder.mkdir(parents=True, exist_ok=True)
    pending = []
    for upload in uploads:
        name = upload.name
        if (not name or any(c in name for c in '/\\:') or name.startswith('.') or
                Path(name).suffix.lower() not in EXTENSIONS or
                re.match(r"^(CON|PRN|AUX|NUL|COM[1-9]|LPT[1-9])(?:\.|$)", name, re.I)):
            raise ValueError("不支持的文件名：" + name)
        target = folder/name
        if target.is_symlink() or not target.resolve().is_relative_to(folder):
            raise ValueError("上传目标不安全")
        if target.exists() and not overwrite:
            raise ValueError(f"{name} 已存在，请勾选允许覆盖或更换文件名")
        data = upload.getvalue()
        if len(data) > MAX_BYTES:
            raise ValueError(f"{name} 超过 20 MB")
        pending.append((target, data))
    for target, data in pending:
        target.write_bytes(data)
    return len(pending)


def main():
    st.set_page_config(page_title="项目文档问答", page_icon="📚", layout="wide")
    st.title("项目文档知识库问答")
    st.caption("导入项目说明，检索原文，用可核对的来源回答问题。")
    with st.sidebar:
        st.header("知识库")
        demo = st.toggle("离线演示（无需 API Key）", value=True)
        name = st.text_input("知识库名称（英文、数字、下划线）", "default")
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,40}", name):
            st.error("知识库名称格式不正确")
            st.stop()
        config = Config.load(APP/".rag-data"/(name+ ("-demo" if demo else "-live")), demo)
        identity = hashlib.sha256((json.dumps(config.fingerprint(), sort_keys=True)+str(config.data_dir)+
                                  config.llm_key+config.llm_model+config.llm_url+config.embedding_key).encode()).hexdigest()
        if st.session_state.get("identity") != identity:
            st.session_state.identity = identity
            st.session_state.index = Index(config)
            st.session_state.chat = []
        index = st.session_state.index
        # Refresh non-index settings after .env/parameter changes.
        index.config = config
        root = st.text_input("文档目录", str(APP/"sample_docs"))
        uploads = st.file_uploader("导入 MD / TXT / PDF", type=["md", "txt", "pdf"], accept_multiple_files=True)
        overwrite = st.checkbox("上传时允许覆盖同名文件", value=False)
        if st.button("保存上传文件到文档目录", disabled=not uploads):
            st.success(f"已保存 {upload_files(root, uploads, overwrite)} 个文件；请点击同步索引。")
        rebuild = st.checkbox("完整重建（模型或分块参数变化时使用）", value=False)
        st.caption("同步会更新新增/修改文件，并移除已删除源文件的索引。一个知识库绑定一个目录。")
        if st.button("同步 / 更新索引", type="primary"):
            with st.spinner("解析文档并建立索引；首次本地 Embedding 可能需要下载模型……"):
                result = index.update(root, rebuild)
            st.session_state.chat = []
            st.success("索引同步完成")
            st.json(result)
        status = index.status()
        st.caption(f"文件 {status['files']} · 片段 {status['chunks']} · 版本 {status.get('revision', 0)}")
        mode = st.selectbox("检索方式", ["hybrid", "vector", "keyword"],
            format_func=lambda x: {"hybrid":"混合检索", "vector":"向量检索", "keyword":"关键词检索"}[x])
        retrieve_only = st.checkbox("只看检索结果，不调用聊天模型", value=False)
        multi_turn = st.checkbox("使用最近对话理解追问（额外一次模型调用）", value=False, disabled=demo)
        if st.button("清空聊天"):
            st.session_state.chat = []
        st.caption("本地单用户应用；退出请在启动终端按 Ctrl+C。")
    if demo:
        st.info("离线演示：使用词法哈希向量，展示检索原文，不生成 LLM 答案；不能据此评估语义问答质量。")
    else:
        st.caption(f"回答模型：{config.llm_model} · Embedding：{config.embedding_model}")
    for turn in st.session_state.chat:
        with st.chat_message("user"):
            st.write(turn["question"])
        with st.chat_message("assistant"):
            st.write(turn["answer"])
            show_sources(turn.get("sources", []))
    question = st.chat_input("例如：如何启动示例项目？API Key 配置在哪里？")
    if question:
        with st.chat_message("user"):
            st.write(question)
        with st.chat_message("assistant"), st.spinner("检索中……"):
            if retrieve_only:
                result = search(index, question, mode)
                result.update(question=question, sources=result["hits"], answer="仅展示检索结果，未调用聊天模型。", status="retrieved")
            else:
                history = [dict(question=t["question"], answer=t["answer"]) for t in st.session_state.chat[-4:]] if multi_turn else None
                result = RAG(index).ask(question, mode, history)
            st.write(result["answer"])
            show_sources(result["sources"])
            st.caption(f"状态：{result['status']} · 用时 {result.get('seconds', '—')} 秒 · {result.get('usage', {})}")
            st.session_state.chat.append(result)
    if st.session_state.chat:
        st.download_button("导出本轮聊天与引用 JSON", json.dumps(st.session_state.chat, ensure_ascii=False, indent=2),
                           file_name="rag-conversation.json", mime="application/json")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        import os
        text = str(exc)
        for key in (os.getenv("LLM_API_KEY"), os.getenv("EMBEDDING_API_KEY")):
            if key:
                text = text.replace(key, "[REDACTED]")
        st.error(text)
        st.info("请检查 .env、依赖和索引配置。排错步骤见 README；修改密钥后重启应用。")
