import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1] / "src"


def test_content_never_imports_other_subsystems():
    imports = set()
    for source in ROOT.rglob("*.py"):
        tree = ast.parse(source.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                imports.add(node.module)
            elif isinstance(node, ast.Import):
                imports.update(alias.name for alias in node.names)
    assert not {name for name in imports if name.startswith("stock_agent") or name.startswith("stock_factor")}
