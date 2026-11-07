import ast
import io
from pathlib import Path
import tokenize
import unittest


ROOT = Path(__file__).resolve().parents[1]


class ContractTests(unittest.TestCase):
    def test_required_modules_exist(self):
        for name in ("config", "physics", "preprocessing", "model", "inference", "metrics"):
            with self.subTest(module=name):
                self.assertTrue((ROOT / "src" / "neural_holography" / f"{name}.py").is_file())

    def test_python_contains_only_code(self):
        for folder in ("src", "scripts", "tests"):
            for path in (ROOT / folder).rglob("*.py"):
                with self.subTest(path=path):
                    source = path.read_text(encoding="utf-8")
                    tree = ast.parse(source)
                    self.assertFalse(any(isinstance(n, ast.Assert) for n in ast.walk(tree)))
                    self.assertFalse(any(isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "print" for n in ast.walk(tree)))
                    self.assertFalse(any(t.type == tokenize.COMMENT for t in tokenize.generate_tokens(io.StringIO(source).readline)))
                    for node in ast.walk(tree):
                        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                            self.assertIsNone(ast.get_docstring(node))
