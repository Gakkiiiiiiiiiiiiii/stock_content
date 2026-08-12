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


def test_domain_has_no_framework_or_infrastructure_imports():
    forbidden = ("fastapi", "sqlalchemy", "qdrant_client", "httpx")
    violations = []
    for source in (ROOT / "stock_content" / "domain").rglob("*.py"):
        tree = ast.parse(source.read_text(encoding="utf-8"))
        names = []
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                names.append(node.module)
            elif isinstance(node, ast.Import):
                names.extend(alias.name for alias in node.names)
        violations.extend(f"{source.name}: {name}" for name in names if name.startswith(forbidden))
    assert not violations
