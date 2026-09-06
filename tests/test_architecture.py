import ast
import re
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "src"


def _application_imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    imports = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("stock_content.application"):
            imports.add(node.module)
        elif isinstance(node, ast.Import):
            imports.update(
                alias.name for alias in node.names if alias.name.startswith("stock_content.application")
            )
    return imports


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


def test_content_integrates_quant_only_via_http():
    # §6.1：content 对 quant 只能走 HTTP（market-data.v1），禁止 import quant 的 Python 实现。
    imports = set()
    for source in ROOT.rglob("*.py"):
        tree = ast.parse(source.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                imports.add(node.module)
            elif isinstance(node, ast.Import):
                imports.update(alias.name for alias in node.names)
    assert not {name for name in imports if name == "quant_demo" or name.startswith("quant_demo.")}


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


def test_application_pipeline_boundaries_do_not_reverse_dependencies():
    application = ROOT / "stock_content" / "application"
    stage_implementations = _application_imports(application / "stages.py")
    replay_facade = _application_imports(application / "replay_service.py")
    replay_implementation = {
        imported
        for source in (application / "replay").rglob("*.py")
        for imported in _application_imports(source)
    }
    verification_facade = _application_imports(application / "verification_refresh.py")
    verification_implementation = {
        imported
        for source in (
            application / "verification" / "__init__.py",
            application / "verification" / "closure.py",
            application / "verification" / "persistence.py",
            application / "verification" / "transaction.py",
        )
        for imported in _application_imports(source)
    }
    verification_stage_facade = _application_imports(application / "verification" / "stages.py")

    # Stages are leaf implementations. Replay and verification may use the
    # pipeline protocol, but neither may reach back into orchestration or
    # another domain's concrete stages.
    assert not {
        imported
        for imported in stage_implementations
        if imported in {
            "stock_content.application.service",
            "stock_content.application.replay_service",
            "stock_content.application.verification_refresh",
        }
        or imported.startswith("stock_content.application.replay")
        or imported.startswith("stock_content.application.verification")
    }
    assert not {
        imported
        for imported in replay_facade | replay_implementation
        if imported in {
            "stock_content.application.service",
            "stock_content.application.stages",
            "stock_content.application.verification_refresh",
        }
        or imported.startswith("stock_content.application.verification")
    }
    assert not {
        imported
        for imported in verification_facade | verification_implementation
        if imported in {
            "stock_content.application.service",
            "stock_content.application.stages",
            "stock_content.application.replay_service",
        }
        or imported.startswith("stock_content.application.replay")
    }
    assert verification_stage_facade == {"stock_content.application.stages"}


def test_application_entrypoints_depend_on_public_pipeline_facades_only():
    application = ROOT / "stock_content" / "application"
    service_imports = _application_imports(application / "service.py")
    entrypoint_imports = (
        _application_imports(ROOT / "stock_content" / "api" / "main.py")
        | _application_imports(ROOT / "stock_content" / "workers" / "content_worker.py")
        | _application_imports(ROOT / "stock_content" / "workers" / "signal_publisher_worker.py")
        | _application_imports(ROOT / "stock_content" / "workers" / "verification_worker.py")
    )
    internal_modules = (
        "stock_content.application.replay.",
        "stock_content.application.verification.",
    )

    # ContentApplication orchestrates through the stable service façades; API
    # and worker entrypoints must not bypass those boundaries.
    assert "stock_content.application.replay_service" in service_imports
    assert "stock_content.application.verification_service" in service_imports
    assert not {
        imported for imported in service_imports if imported.startswith(internal_modules)
    }
    assert not {
        imported for imported in entrypoint_imports if imported.startswith(internal_modules)
    }


def _installed_extras(path: Path) -> set[str]:
    match = re.search(r'(?:-e\s+)?"?\.\[([^]]+)\]"?', path.read_text(encoding="utf-8"))
    assert match, f"missing extra install in {path}"
    return {name.strip() for name in match.group(1).split(",")}


def test_core_images_and_runtime_exclude_heavy_media_dependencies():
    project = tomllib.loads((ROOT.parent / "pyproject.toml").read_text(encoding="utf-8"))
    extras = project["project"]["optional-dependencies"]
    heavy_profiles = {"media-cpu", "ocr", "diarization", "multimodal-gpu", "media", "multimodal"}
    heavy_modules = {"faster_whisper", "yt_dlp", "opencc", "paddleocr", "pyannote", "torch"}

    assert not extras["core"]
    for dockerfile in ("Dockerfile", "docker/Dockerfile.api", "docker/Dockerfile.core-worker"):
        assert not (_installed_extras(ROOT.parent / dockerfile) & heavy_profiles)

    violations = []
    for source in (
        ROOT / "stock_content" / "api" / "main.py",
        ROOT / "stock_content" / "api" / "dependencies.py",
        ROOT / "stock_content" / "workers" / "content_worker.py",
        ROOT / "stock_content" / "workers" / "verification_worker.py",
        ROOT / "stock_content" / "workers" / "signal_publisher_worker.py",
    ):
        tree = ast.parse(source.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            names = ([node.module] if isinstance(node, ast.ImportFrom) and node.module else [])
            if isinstance(node, ast.Import):
                names.extend(alias.name for alias in node.names)
            violations.extend(f"{source.name}: {name}" for name in names if name.split(".")[0] in heavy_modules)
    assert not violations


def test_specialized_worker_profiles_match_their_auditable_dependency_sets():
    project = tomllib.loads((ROOT.parent / "pyproject.toml").read_text(encoding="utf-8"))
    extras = project["project"]["optional-dependencies"]
    assert {"faster-whisper", "yt-dlp", "opencc-python-reimplemented", "paddleocr"} <= {
        requirement.split(">=", 1)[0] for requirement in extras["media"]
    }
    assert {"faster-whisper", "yt-dlp", "opencc-python-reimplemented", "paddleocr", "pyannote.audio"} <= {
        requirement.split(">=", 1)[0] for requirement in extras["multimodal"]
    }

    for profile in ("media", "multimodal"):
        expected = f"-e .[core,postgres,search,{profile}]"
        assert expected in (ROOT.parent / "requirements" / f"{profile}.in").read_text(encoding="utf-8")
        assert expected in (ROOT.parent / "locks" / f"{profile}.lock").read_text(encoding="utf-8")
        assert _installed_extras(ROOT.parent / "docker" / f"Dockerfile.{profile}") == {
            "core", "postgres", "search", profile
        }
