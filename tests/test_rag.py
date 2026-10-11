import io
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

from rag.answer import RAG, make_llm
from rag.config import APP, Config
from rag.embedding import DemoEmbeddings, make_embeddings
from rag.ingest import chunks
from rag.retrieve import search
from rag.store import Index


class CountingEmbeddings(DemoEmbeddings):
    def __init__(self):
        self.count = 0

    def embed_documents(self, texts):
        self.count += len(texts)
        return super().embed_documents(texts)


class RAGTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.docs = self.root/"docs"
        self.docs.mkdir()
        self.config = Config(data_dir=self.root/"db", provider="demo", embedding_model="hash-demo-v1", demo=True, min_cosine=0.05)
        self.emb = CountingEmbeddings()
        self.index = Index(self.config, self.emb)

    def tearDown(self):
        self.temp.cleanup()

    def seed(self):
        (self.docs/"config.md").write_text("# 配置\nLLM_API_KEY 在 .env 文件中配置。\n默认端口 8501。\n", encoding="utf-8")
        (self.docs/"storage.txt").write_text("SQLite 使用事务保存索引；失败时保留旧数据。", encoding="utf-8")
        return self.index.update(self.docs)

    def live(self, value):
        # Only generation is real-mode here; fixed embeddings are injected explicitly.
        self.config.demo = False
        return RAG(self.index, RunnableLambda(lambda _: AIMessage(content=json.dumps(value, ensure_ascii=False))))

    def test_persistent_index_and_all_retrieval_modes(self):
        self.seed()
        restored = Index(self.config, self.emb)
        for mode in ("vector", "keyword", "hybrid"):
            hits = search(restored, "LLM_API_KEY", mode)["hits"]
            self.assertEqual(hits[0]["source"], "config.md")
        self.assertEqual(restored.status()["files"], 2)

    def test_incremental_change_delete_and_no_reembedding(self):
        self.seed()
        before = self.emb.count
        self.assertEqual(self.index.update(self.docs)["embedded_chunks"], 0)
        self.assertEqual(self.emb.count, before)
        (self.docs/"config.md").write_text("NEW_PORT=9900", encoding="utf-8")
        (self.docs/"storage.txt").unlink()
        report = self.index.update(self.docs)
        self.assertEqual((report["changed"], report["removed"]), (1, 1))
        _, docs, _ = self.index.snapshot()
        self.assertEqual(len(docs), 1)
        self.assertIn("9900", docs[0].page_content)
        self.assertEqual(search(self.index, "SQLite", "keyword")["hits"], [])

    def test_embedding_failure_keeps_entire_old_index(self):
        self.seed()
        revision = self.index.status()["revision"]
        (self.docs/"config.md").write_text("changed contents", encoding="utf-8")
        with patch.object(self.emb, "embed_documents", side_effect=RuntimeError("provider unavailable")):
            with self.assertRaises(RuntimeError):
                self.index.update(self.docs, rebuild=True)
        self.assertEqual(self.index.status()["revision"], revision)
        self.assertIn("LLM_API_KEY", self.index.snapshot()[1][0].page_content + self.index.snapshot()[1][1].page_content)

    def test_parse_failure_keeps_old_index(self):
        self.seed()
        (self.docs/"broken.pdf").write_bytes(b"not a pdf")
        with self.assertRaises(Exception):
            self.index.update(self.docs)
        self.assertEqual(self.index.status()["files"], 2)

    def test_config_mismatch_requires_rebuild(self):
        self.seed()
        self.config.chunk_size = 500
        with self.assertRaisesRegex(ValueError, "重建"):
            self.index.snapshot()
        self.index.update(self.docs, rebuild=True)
        self.assertTrue(self.index.snapshot()[1])

    def test_different_document_root_rejected(self):
        self.seed()
        other = self.root/"other"
        other.mkdir()
        with self.assertRaisesRegex(ValueError, "另一个目录"):
            self.index.update(other, rebuild=True)

    def test_exact_source_offsets_and_line_numbers(self):
        self.config.chunk_size = 100
        self.config.overlap = 20
        text = "第一行\n" + ("重复段落：配置项测试。\n"*30)
        docs = chunks(text.encode(), "重复.md", self.config)
        self.assertGreater(len(docs), 2)
        for doc in docs:
            pos = doc.metadata["start"]
            self.assertEqual(text[pos:pos+len(doc.page_content)], doc.page_content)
            self.assertEqual(doc.metadata["line_start"], text[:pos].count("\n")+1)

    def test_scanned_pdf_rejected(self):
        from pypdf import PdfWriter
        writer = PdfWriter()
        writer.add_blank_page(width=100, height=100)
        buf = io.BytesIO()
        writer.write(buf)
        with self.assertRaisesRegex(ValueError, "OCR"):
            chunks(buf.getvalue(), "scan.pdf", self.config)

    def test_text_pdf_page_citation(self):
        from pypdf import PdfWriter
        from pypdf.generic import DictionaryObject, NameObject, DecodedStreamObject
        writer = PdfWriter()
        page = writer.add_blank_page(width=500, height=500)
        font = DictionaryObject({NameObject("/Type"):NameObject("/Font"), NameObject("/Subtype"):NameObject("/Type1"),
                                 NameObject("/BaseFont"):NameObject("/Helvetica")})
        page[NameObject("/Resources")] = DictionaryObject({NameObject("/Font"):DictionaryObject({NameObject("/F1"):writer._add_object(font)})})
        stream = DecodedStreamObject()
        stream.set_data(b"BT /F1 12 Tf 10 450 Td (Port 8501) Tj ET")
        page[NameObject("/Contents")] = writer._add_object(stream)
        buf = io.BytesIO()
        writer.write(buf)
        docs = chunks(buf.getvalue(), "manual.pdf", self.config)
        self.assertEqual(docs[0].metadata["page"], 1)
        self.assertIn("8501", docs[0].page_content)

    def test_hidden_documents_are_excluded(self):
        (self.docs/".private.md").write_text("secret")
        (self.docs/"ok.txt").write_text("public")
        self.index.update(self.docs)
        self.assertEqual(self.index.status()["files"], 1)

    def test_no_evidence_does_not_call_llm(self):
        self.seed()
        def never(_):
            self.fail("LLM should not be called")
        self.config.demo = False
        result = RAG(self.index, RunnableLambda(never)).ask("unrelatedxyz987", mode="keyword")
        self.assertEqual(result["status"], "no_evidence")
        self.assertEqual(result["usage"]["llm_calls"], 0)

    def test_valid_answer_and_citations(self):
        self.seed()
        result = self.live(dict(supported=True, answer="在 .env 中配置。[1]", citations=[1])).ask("LLM_API_KEY", "keyword")
        self.assertEqual(result["status"], "answered")
        self.assertEqual(result["sources"][0]["source"], "config.md")
        self.assertEqual(result["usage"]["llm_calls"], 1)

    def test_invalid_citation_is_not_presented(self):
        self.seed()
        for ids, text in (([99], "伪造回答 [99]"), ([], "没有引用"), ([1], "正文是 [2]")):
            result = self.live(dict(supported=True, answer=text, citations=ids)).ask("LLM_API_KEY", "keyword")
            self.assertEqual(result["status"], "invalid_answer")
            self.assertNotIn(text, result["answer"])

    def test_model_can_abstain_despite_hits(self):
        self.seed()
        result = self.live(dict(supported=False, answer="现有资料没有该信息。", citations=[])).ask("LLM_API_KEY", "keyword")
        self.assertEqual(result["status"], "abstained")

    def test_demo_does_not_claim_generated_answer(self):
        self.seed()
        result = RAG(self.index).ask("LLM_API_KEY")
        self.assertEqual(result["status"], "demo")
        self.assertFalse(result["supported"])
        self.assertEqual(result["usage"]["llm_calls"], 0)

    def test_multiturn_rewrite_then_grounded_generation(self):
        self.seed()
        self.config.demo = False
        seen = []
        replies = iter(["LLM_API_KEY 配置在哪里？", '{"supported":true,"answer":"在 .env 中配置。[1]","citations":[1]}'])
        def model(prompt):
            seen.append(prompt.to_string())
            return AIMessage(content=next(replies))
        result = RAG(self.index, RunnableLambda(model)).ask("它在哪配置？", "keyword",
            history=[{"question":"LLM_API_KEY 是什么？", "answer":"模型密钥"}])
        self.assertEqual(result["query"], "LLM_API_KEY 配置在哪里？")
        self.assertEqual(result["usage"]["llm_calls"], 2)
        self.assertEqual(result["status"], "answered")
        self.assertNotIn("模型密钥", seen[1])

    def test_truncated_generation_is_not_accepted(self):
        self.seed()
        self.config.demo = False
        model = RunnableLambda(lambda _: AIMessage(content='{"supported":true}', response_metadata={"finish_reason":"length"}))
        result = RAG(self.index, model).ask("LLM_API_KEY", "keyword")
        self.assertEqual(result["status"], "invalid_answer")

    def test_context_budget_limits_exposed_sources(self):
        self.seed()
        self.config.context_chars = 200
        result = RAG(self.index).ask("配置 SQLite", "keyword")
        self.assertLessEqual(sum(len(s["text"])+len(s["source"])+100 for s in result["sources"]), 200)
        self.assertEqual([s["citation"] for s in result["sources"]], list(range(1,len(result["sources"])+1)))

    def test_concurrent_sync_serializes(self):
        self.seed()
        (self.docs/"new.md").write_text("new documentation", encoding="utf-8")
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(2) as pool:
            results = list(pool.map(lambda _: Index(self.config, self.emb).update(self.docs), range(2)))
        self.assertEqual(sum(r["changed"] for r in results), 1)
        self.assertEqual(self.index.status()["files"], 3)

    def test_invalid_vector_rolls_back(self):
        self.seed()
        (self.docs/"new.md").write_text("new")
        with patch.object(self.emb, "embed_documents", return_value=[[float("nan")]]):
            with self.assertRaises(ValueError):
                self.index.update(self.docs)
        self.assertEqual(self.index.status()["files"], 2)


class HTTPIntegrationTests(unittest.TestCase):
    def test_real_langchain_embedding_and_chat_clients(self):
        received = []
        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                received.append((self.path, body))
                if self.path.endswith("/embeddings"):
                    items = body["input"]
                    if isinstance(items, str):
                        items = [items]
                    value = {"object":"list", "data":[{"object":"embedding", "index":i, "embedding":[1.,0.,0.]} for i in range(len(items))],
                             "model":"fixture", "usage":{"prompt_tokens":3,"total_tokens":3}}
                else:
                    value = {"id":"fixture", "object":"chat.completion", "created":1,"model":"fixture",
                        "choices":[{"index":0,"finish_reason":"stop", "message":{"role":"assistant",
                        "content":json.dumps({"supported":True,"answer":"端口是 8501。[1]","citations":[1]}, ensure_ascii=False)}}],
                        "usage":{"prompt_tokens":20,"completion_tokens":10,"total_tokens":30}}
                payload = json.dumps(value).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
            def log_message(self, *args):
                pass
        server = ThreadingHTTPServer(("127.0.0.1",0), Handler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        try:
            with tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                docs = root/"docs"
                docs.mkdir()
                (docs/"port.md").write_text("端口是 8501。", encoding="utf-8")
                url = f"http://127.0.0.1:{server.server_port}/v1"
                config = Config(data_dir=root/"db", provider="openai", embedding_model="fixture", embedding_url=url,
                                embedding_key="test-only", llm_key="test-only", llm_url=url, llm_model="fixture")
                index = Index(config)
                index.update(docs)
                result = RAG(index).ask("端口是多少？")
                self.assertEqual(result["status"], "answered")
                self.assertEqual(result["usage"]["total_tokens"], 30)
                self.assertTrue(any(path.endswith("/embeddings") for path,_ in received))
                self.assertTrue(any(path.endswith("/chat/completions") for path,_ in received))
                chat_body = next(b for path,b in received if path.endswith("/chat/completions"))
                self.assertEqual(chat_body["max_tokens"], config.max_tokens)
                self.assertNotIn("max_completion_tokens", chat_body)
                self.assertTrue(all(isinstance(x,str) for path,b in received if path.endswith("/embeddings") for x in b["input"]))
        finally:
            server.shutdown()
            server.server_close()


if __name__ == "__main__":
    unittest.main()
