import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from rag.config import APP


class CLITests(unittest.TestCase):
    def test_index_search_and_evaluation_commands(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            def run(*args):
                process = subprocess.run([sys.executable, "-m", "rag", "--demo", "--data-dir", str(root/"db"), *args],
                    cwd=APP, capture_output=True, text=True, encoding="utf-8", timeout=30)
                self.assertEqual(process.returncode, 0, process.stderr)
                return json.loads(process.stdout)
            self.assertEqual(run("index", str(APP/"sample_docs"))["changed"], 5)
            result = run("search", "LLM_API_KEY", "--mode", "keyword")
            self.assertEqual(result["hits"][0]["source"], "configuration.md")
            run("evaluate", "--output", str(root/"report.json"))
            report = json.loads((root/"report.json").read_text(encoding="utf-8"))
            self.assertEqual(report["top_k"], 3)
            self.assertEqual(set(report["modes"]), {"vector", "keyword", "hybrid"})
            self.assertEqual(report["modes"]["hybrid"]["questions"], 22)
            self.assertTrue(report["demo"])


if __name__ == "__main__":
    unittest.main()
