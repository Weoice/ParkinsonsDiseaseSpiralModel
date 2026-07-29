"""Assert Vercel can find a valid Python handler export."""
from pathlib import Path
import ast


def test_api_predict_exports_handler():
    src = (Path(__file__).resolve().parents[1] / "api" / "predict.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    names = {
        node.name
        for node in tree.body
        if isinstance(node, ast.ClassDef) or isinstance(node, ast.FunctionDef)
    }
    assigns = set()
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    assigns.add(target.id)
    assert "handler" in names or "app" in assigns or "application" in assigns


def test_no_root_app_py():
    root = Path(__file__).resolve().parents[1]
    assert not (root / "app.py").exists(), "Root app.py confuses Vercel; use streamlit_app.py locally"
