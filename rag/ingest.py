import hashlib
from io import BytesIO
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

EXTENSIONS = {".md", ".txt", ".pdf"}
SKIP = {".git", ".venv", "venv", "node_modules", "__pycache__", ".rag-data"}
MAX_BYTES = 20 * 1024 * 1024
MAX_FILES = 2000


def scan(root):
    import os
    root = Path(root).resolve()
    if not root.is_dir():
        raise ValueError("文档目录不存在")
    files = []
    for parent, dirs, names in os.walk(root, followlinks=False):
        dirs[:] = sorted(d for d in dirs if not d.startswith(".") and d not in SKIP
                         and not (Path(parent)/d).is_symlink())
        for name in sorted(names):
            path = Path(parent)/name
            if path.suffix.lower() not in EXTENSIONS or name.startswith(".") or path.is_symlink():
                continue
            if not path.resolve().is_relative_to(root):
                continue
            files.append(path)
            if len(files) > MAX_FILES:
                raise ValueError(f"每个库最多 {MAX_FILES} 个文档，请缩小范围")
    return root, files


def read_bytes(path):
    with path.open("rb") as stream:
        data = stream.read(MAX_BYTES+1)
    if len(data) > MAX_BYTES:
        raise ValueError(f"{path.name} 超过 20 MB，未更新索引")
    return data


def load_pages(data, source):
    if Path(source).suffix.lower() == ".pdf":
        from pypdf import PdfReader
        reader = PdfReader(BytesIO(data))
        if reader.is_encrypted:
            raise ValueError(f"{source} 是加密 PDF")
        if len(reader.pages) > 500:
            raise ValueError(f"{source} 超过 500 页")
        pages = [(page.extract_text() or "", {"source": source, "page": i+1})
                 for i, page in enumerate(reader.pages)]
        if not any(text.strip() for text, _ in pages):
            raise ValueError(f"{source} 没有可提取文本；扫描 PDF 需要先 OCR")
        return pages
    text = data.decode("utf-8-sig").replace("\r\n", "\n").replace("\r", "\n")
    return [(text, {"source": source, "page": 0})]


def chunks(data, source, config):
    splitter = RecursiveCharacterTextSplitter(chunk_size=config.chunk_size,
        chunk_overlap=config.overlap, add_start_index=True, strip_whitespace=False,
        separators=["\n\n", "\n", "。", "！", "？", ". ", " ", ""])
    output = []
    for text, meta in load_pages(data, source):
        docs = splitter.create_documents([text], [meta])
        for doc in docs:
            if not doc.page_content.strip():
                continue
            offset = doc.metadata.pop("start_index")
            if offset < 0 or text[offset:offset+len(doc.page_content)] != doc.page_content:
                raise ValueError("无法准确定位文档片段，拒绝生成错误引用")
            doc.metadata.update(start=offset, line_start=text.count("\n", 0, offset)+1,
                line_end=text.count("\n", 0, offset+len(doc.page_content.rstrip("\n")))+1)
            ident = f"{source}\0{meta['page']}\0{offset}\0{doc.page_content}"
            doc.metadata["id"] = hashlib.sha256(ident.encode()).hexdigest()
            output.append(doc)
    return output
