from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from streamlit.testing.v1 import AppTest

from rag.config import APP, Config


class UITests(unittest.TestCase):
    def test_index_chat_and_source_panel(self):
        with tempfile.TemporaryDirectory() as temp:
            config = Config(data_dir=Path(temp)/"db", provider="demo", embedding_model="hash-demo-v1", demo=True)
            with patch("rag.config.Config.load", return_value=config):
                app = AppTest.from_file(str(APP/"app.py"), default_timeout=30).run()
                self.assertEqual(len(app.exception), 0)
                button = next(b for b in app.button if b.label == "同步 / 更新索引")
                button.click().run()
                self.assertEqual(len(app.exception), 0)
                self.assertEqual(len(app.error), 0)
                app.chat_input[0].set_value("LLM_API_KEY 在哪里配置？").run()
                self.assertEqual(len(app.exception), 0)
                self.assertEqual(len(app.error), 0)
                self.assertTrue(app.expander)
                self.assertIn("离线检索演示", app.session_state["chat"][-1]["answer"])

    def test_upload_path_escape_and_overwrite_protection(self):
        from app import upload_files
        class Upload:
            def __init__(self, name): self.name = name
            def getvalue(self): return b"hello"
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for name in ("../escape.md", "C:\\bad.md", ".env", "CON.txt"):
                with self.assertRaises(ValueError):
                    upload_files(root, [Upload(name)])
            self.assertEqual(upload_files(root, [Upload("ok.md")]), 1)
            with self.assertRaises(ValueError):
                upload_files(root, [Upload("ok.md")])


if __name__ == "__main__":
    unittest.main()
