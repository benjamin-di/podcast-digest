"""Offline build provenance, sink-preservation and publisher acceptance checks."""
from __future__ import annotations

import importlib.util
import io
import json
import logging
import os
import shutil
import subprocess
import sys
import tarfile
import tomllib
import types
import zipfile
from dataclasses import asdict
from pathlib import Path

import pytest
from local_transcriber import identity as transcriber_identity

from podcast_digest import identity, log

ROOT = Path(__file__).resolve().parents[1]
REVISION = "a" * 40
PODCAST_VERSION = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["version"]
TRANSCRIBER_VERSION = tomllib.loads(
    (ROOT / "packages/local-transcriber/pyproject.toml").read_text())["project"]["version"]


def script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / name)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def project(tmp_path):
    for rel, version in [("", "0.4.7"), ("packages/local-transcriber", "0.2.0")]:
        base = tmp_path / rel
        base.mkdir(parents=True, exist_ok=True)
        (base / "pyproject.toml").write_text(f'[project]\nversion = "{version}"\n')
    for rel in ["src/podcast_digest", "src/podcast_digest_mcp",
                "packages/local-transcriber/src/local_transcriber"]:
        (tmp_path / rel).mkdir(parents=True)
    return tmp_path


@pytest.mark.parametrize("revision", ["a" * 40, "B" * 64])
def test_embed_separate_products_and_services(tmp_path, revision):
    root = project(tmp_path)
    embed = script("embed-build-identity.py").embed
    embed(root, "podcast-digest", "v0.4.7", revision)
    embed(root, "local-transcriber", "0.2.0", revision)
    for path, service, version in [
        ("src/podcast_digest", "podcast-digest", "0.4.7"),
        ("src/podcast_digest_mcp", "podcast-digest-mcp", "0.4.7"),
        ("packages/local-transcriber/src/local_transcriber", "local-transcriber", "0.2.0"),
    ]:
        assert json.loads((root / path / "build_identity.json").read_text()) == {
            "service": service, "version": version, "source_revision": revision.lower()}


@pytest.mark.parametrize("product,version,revision", [
    ("podcast-digest", "", REVISION), ("podcast-digest", "0.2.0", REVISION),
    ("local-transcriber", "0.4.7", REVISION), ("podcast-digest", "0.4.7", "unknown"),
    ("podcast-digest", "0.4.7", "a" * 39), ("podcast-digest", "0.4.7", "g" * 40),
    ("podcast-digest", "0.4.7", "a" * 41), ("podcast-digest", "0.4.7", "a" * 65),
    ("podcast-digest", "0.4.7", " " + REVISION), ("foreign", "0.4.7", REVISION),
])
def test_embed_rejects_bad_inputs_before_writing(tmp_path, product, version, revision):
    root = project(tmp_path)
    with pytest.raises(ValueError):
        script("embed-build-identity.py").embed(root, product, version, revision)
    assert not list(root.rglob("build_identity.json"))


@pytest.fixture(params=[identity, transcriber_identity])
def reader(request):
    return request.param


def test_missing_installed_stamp_requires_explicit_editable(reader, tmp_path, monkeypatch):
    missing = tmp_path / "absent.json"
    for direct in [None, "{}", "not json", "[]", '{"dir_info":{"editable":false}}',
                   '{"dir_info":{"editable":"true"}}']:
        monkeypatch.setattr(reader, "distribution", lambda name, direct=direct: types.SimpleNamespace(read_text=lambda _: direct))
        with pytest.raises(ValueError, match="Missing"):
            reader.load_build(missing)
    monkeypatch.setattr(reader, "distribution", lambda name: types.SimpleNamespace(
        read_text=lambda _: '{"dir_info":{"editable":true}}'))
    assert reader.load_build(missing).source_revision == "unknown"


def test_source_unknown_is_explicit_and_environment_cannot_relabel(reader, tmp_path, monkeypatch):
    def absent(name):
        raise reader.PackageNotFoundError(name)
    monkeypatch.setattr(reader, "distribution", absent)
    monkeypatch.setenv("VERSION", "99.0")
    monkeypatch.setenv("VCS_REF", "b" * 64)
    monkeypatch.setenv("LOCAL_TRANSCRIBER_SOURCE_REV", "b" * 40)
    assert reader.load_build(tmp_path / "absent").source_revision == "unknown"
    path = tmp_path / "stamp.json"
    expected = {**asdict(reader.BUILD), "source_revision": REVISION}
    path.write_text(json.dumps(expected))
    assert asdict(reader.load_build(path)) == expected


@pytest.mark.parametrize("change", [
    {"service": "foreign"}, {"version": "99.0"}, {"source_revision": "abc123"},
    {"source_revision": "B" * 40}, {"source_revision": 3}, {"extra": "no"},
])
def test_loader_rejects_conflicting_or_malformed_stamp(reader, tmp_path, change):
    path = tmp_path / "stamp.json"
    path.write_text(json.dumps({**asdict(reader.BUILD), "source_revision": REVISION, **change}))
    with pytest.raises(ValueError):
        reader.load_build(path)


@pytest.mark.parametrize("payload", ['[]', '{}', '{', '{"service":1,"service":2}', '\ufffd'])
def test_loader_rejects_invalid_shape_and_duplicates(reader, tmp_path, payload):
    path = tmp_path / "stamp.json"
    path.write_text(payload)
    with pytest.raises(ValueError):
        reader.load_build(path)


GIT = ["git", "-c", "user.name=synthetic", "-c", "user.email=synthetic@example.com", "-c", "commit.gpgsign=false"]
TRANSCRIBER = "packages/local-transcriber"


def git(*args, cwd):
    return subprocess.run([*GIT, *args], cwd=cwd, check=True, capture_output=True, text=True).stdout.strip()


def source_tree(target):
    """The packaging inputs of this checkout, without generated stamps or caches."""
    skip = shutil.ignore_patterns("__pycache__", "build_identity.json", "tests")
    for name in ["pyproject.toml", "README.md", "LICENSE", "NOTICE", "hatch_build.py", ".gitignore", "scripts"]:
        source = ROOT / name
        if not source.exists():
            continue
        if source.is_dir():
            shutil.copytree(source, target / name, ignore=skip)
        else:
            target.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target / name)
    for name in ["src", TRANSCRIBER]:
        shutil.copytree(ROOT / name, target / name, ignore=skip)
    return target


def committed(target):
    source_tree(target)
    git("init", "-q", "-b", "main", cwd=target)
    git("add", "-A", cwd=target)
    git("commit", "-qm", "synthetic source", cwd=target)
    return git("rev-parse", "HEAD", cwd=target)


def uv_build(project, out, **environment):
    return subprocess.run(["uv", "build", "--offline", "--out-dir", str(out), str(project)],
                          capture_output=True, text=True, timeout=600,
                          env={**os.environ, **environment})


def packaged_stamps(dist):
    found = {}
    for wheel in dist.glob("*.whl"):
        with zipfile.ZipFile(wheel) as archive:
            for name in archive.namelist():
                if name.endswith("/build_identity.json"):
                    found[name] = json.loads(archive.read(name))
    for sdist in dist.glob("*.tar.gz"):
        with tarfile.open(sdist) as archive:
            for member in archive.getmembers():
                if member.name.endswith("/build_identity.json"):
                    found[member.name.split("/", 1)[1]] = json.load(archive.extractfile(member))
    return found


def test_clean_checkout_builds_embed_their_commit_over_stale_and_inherited_identity(tmp_path):
    source = tmp_path / "source"
    revision = committed(source)
    decoy = tmp_path / "decoy"
    git("clone", "-q", str(source), str(decoy), cwd=tmp_path)
    git("commit", "-q", "--allow-empty", "-m", "decoy", cwd=decoy)
    embed = script("embed-build-identity.py").embed
    embed(source, "podcast-digest", PODCAST_VERSION, "b" * 40)
    embed(source, "local-transcriber", TRANSCRIBER_VERSION, "b" * 40)
    for project, stamps in [
        (source / TRANSCRIBER, {
            "local_transcriber/build_identity.json": ("local-transcriber", TRANSCRIBER_VERSION),
            "src/local_transcriber/build_identity.json": ("local-transcriber", TRANSCRIBER_VERSION)}),
        (source, {
            "podcast_digest/build_identity.json": ("podcast-digest", PODCAST_VERSION),
            "podcast_digest_mcp/build_identity.json": ("podcast-digest-mcp", PODCAST_VERSION),
            "src/podcast_digest/build_identity.json": ("podcast-digest", PODCAST_VERSION),
            "src/podcast_digest_mcp/build_identity.json": ("podcast-digest-mcp", PODCAST_VERSION)}),
    ]:
        dist = tmp_path / "dist" / project.name
        # The default build makes the sdist, then the wheel from that sdist.
        result = uv_build(project, dist, GIT_DIR=str(decoy / ".git"), GIT_WORK_TREE=str(decoy))
        assert result.returncode == 0, result.stderr
        assert packaged_stamps(dist) == {
            path: {"service": service, "version": version, "source_revision": revision}
            for path, (service, version) in stamps.items()}


def test_noneditable_builds_without_committed_identity_are_refused(tmp_path):
    dirty = tmp_path / "dirty"
    committed(dirty)
    (dirty / TRANSCRIBER / "src/local_transcriber/uncommitted.py").write_text("VALUE = 1\n")
    result = uv_build(dirty / TRANSCRIBER, tmp_path / "dirty-dist")
    assert result.returncode != 0 and "requires committed source" in result.stderr
    exported = source_tree(tmp_path / "exported")
    result = uv_build(exported / TRANSCRIBER, tmp_path / "exported-dist")
    assert result.returncode != 0 and "Missing build identity" in result.stderr
    assert not list((tmp_path / "exported-dist").glob("*.whl"))


@pytest.mark.parametrize("case", ["local-exclude", "configured-exclude", "unavailable-git", "skip-worktree", "assume-unchanged", "sparse"])
@pytest.mark.parametrize("project", [".", TRANSCRIBER])
def test_source_build_refuses_uncommitted_packaged_files_and_unverifiable_checkout(tmp_path, case, project):
    source = tmp_path / "source"
    committed(source)
    environment = {}
    if case == "unavailable-git":
        embed = script("embed-build-identity.py").embed
        embed(source, "podcast-digest", PODCAST_VERSION, "b" * 40)
        embed(source, "local-transcriber", TRANSCRIBER_VERSION, "b" * 40)
        commands = tmp_path / "bin"
        commands.mkdir()
        (commands / "git").write_text("#!/bin/sh\nexit 128\n")
        (commands / "git").chmod(0o755)
        environment["PATH"] = str(commands) + os.pathsep + os.environ["PATH"]
    elif case in {"skip-worktree", "assume-unchanged", "sparse"}:
        module = "src/podcast_digest/pipeline.py" if project == "." else "src/local_transcriber/whisper.py"
        name = (Path(project) / module).as_posix()
        if case == "sparse":
            git("sparse-checkout", "set", "--no-cone", "/*", f"!/{name}", cwd=source)
        else:
            with (source / name).open("a") as stream:
                stream.write("\nSYNTHETIC_UNCOMMITTED = True\n")
            git("update-index", f"--{case}", name, cwd=source)
        assert git("status", "--porcelain", cwd=source) == ""
    else:
        package = "podcast_digest" if project == "." else "local_transcriber"
        name = (Path(project) / "src" / package / "uncommitted.py").as_posix()
        (source / name).write_text("VALUE = 1\n")
        if case == "local-exclude":
            (source / ".git/info/exclude").write_text(name + "\n")
        else:
            excludes = tmp_path / "machine-excludes"
            excludes.write_text(name + "\n")
            git("config", "core.excludesFile", str(excludes), cwd=source)
    result = uv_build(source / project, tmp_path / "dist", **environment)
    assert result.returncode != 0, result.stderr
    assert not list((tmp_path / "dist").glob("*.whl"))


@pytest.fixture
def clean_logging(monkeypatch):
    loggers = [logging.getLogger(), *(x for x in logging.Logger.manager.loggerDict.values()
                                     if isinstance(x, logging.Logger))]
    states = [(x, x.level, x.propagate, x.handlers[:], [(h, h.formatter) for h in x.handlers]) for x in loggers]
    monkeypatch.setattr(log, "_MCP", False)
    monkeypatch.setattr(logging.getLogger(), "handlers", [])
    yield
    for logger, level, propagate, handlers, formatters in states:
        logger.setLevel(level)
        logger.propagate = propagate
        logger.handlers = handlers
        for handler, formatter in formatters:
            handler.setFormatter(formatter)


@pytest.mark.parametrize("mcp", [False, True])
def test_existing_json_sink_keeps_types_redaction_filters_and_thresholds(clean_logging, mcp):
    class Redactor(logging.Formatter):
        def format(self, record):
            return json.dumps({"message": "safe", "count": 4, "ready": False,
                               "exception": "scrubbed", "logger": record.name})
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(Redactor())
    handler.setLevel(logging.ERROR)
    def filter_(record):
        return record.msg != "drop"
    handler.addFilter(filter_)
    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(logging.WARNING)
    sdk = logging.getLogger("fastmcp.identity_fixture")
    sdk.handlers = [handler]
    sdk.propagate = False
    sdk.setLevel(logging.ERROR)
    log.configure("DEBUG", mcp=mcp)
    log.configure("INFO", mcp=mcp)
    assert root.handlers == [handler]
    assert handler.level == logging.ERROR and root.level == logging.WARNING
    assert handler.filters == [filter_] and handler.stream is stream
    assert not sdk.propagate and sdk.level == logging.ERROR
    sdk.warning("suppressed")
    sdk.error("drop")
    try:
        raise RuntimeError("sensitive exception")
    except RuntimeError:
        sdk.exception("sensitive message", extra={"extras": {"private": "sensitive extra"}})
    payload = json.loads(stream.getvalue())
    assert payload["count"] == 4 and payload["ready"] is False
    assert payload["exception"] == "scrubbed" and payload["message"] == "safe"
    assert "sensitive" not in stream.getvalue()
    assert payload["service"] == ("podcast-digest-mcp" if mcp else "podcast-digest")


def test_worker_tty_multiline_and_child_product_attribution(clean_logging):
    record = logging.LogRecord("podcast_digest.worker", logging.ERROR, "", 0, "first\nsecond", (), None)
    record.extras = {"count": 2, "service": "spoof"}
    rendered = log._KVFormatter("%(asctime)s %(levelname)s %(name)s: %(message)s").format(record)
    assert len(rendered.splitlines()) == 2
    assert all("service=podcast-digest " in line and "instance=" in line for line in rendered.splitlines())
    assert "service_=spoof" in rendered
    child = logging.LogRecord("local_transcriber.whisper", logging.WARNING, "", 0, "safe summary", (), None)
    payload = json.loads(log._JsonFormatter().format(child))
    assert payload["service"] == "local-transcriber" and payload["version"] == TRANSCRIBER_VERSION
    assert payload["instance"] == log.identity_fields()["instance"]


def test_standalone_transcriber_preserves_redacted_sink(clean_logging):
    from local_transcriber import log as transcriber_log
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(logging.Formatter("redacted", validate=False))
    logger = logging.getLogger("local_transcriber.whisper")
    logger.handlers = [handler]
    logger.propagate = False
    logger.setLevel(logging.WARNING)
    transcriber_log.configure()
    logger.warning("sensitive raw output")
    payload = json.loads(stream.getvalue())
    assert payload["msg"] == "redacted" and payload["service"] == "local-transcriber"
    assert "sensitive" not in stream.getvalue()


@pytest.mark.parametrize("variant", ["transcriber", "podcast", "mcp"])
def test_actual_image_acceptance_checks_all_installed_identities(monkeypatch, variant):
    from dataclasses import replace

    from podcast_digest_mcp import identity as mcp_identity
    for module in [identity, transcriber_identity, mcp_identity]:
        monkeypatch.setattr(module, "BUILD", replace(module.BUILD, source_revision=REVISION))
    acceptance = script("check-image-identity.py")
    acceptance.check(variant, REVISION, PODCAST_VERSION, TRANSCRIBER_VERSION)
    with pytest.raises(AssertionError):
        acceptance.check(variant, "b" * 40, PODCAST_VERSION, TRANSCRIBER_VERSION)
    with pytest.raises(AssertionError):
        acceptance.check(variant, REVISION, PODCAST_VERSION, "0.0.0-invalid")
    if variant != "transcriber":
        with pytest.raises(AssertionError):
            acceptance.check(variant, REVISION, "99.0", TRANSCRIBER_VERSION)


def test_actual_mcp_initialize_acceptance_rejects_framework_version(monkeypatch):
    import urllib.request
    code = (ROOT / "scripts/smoke-image.sh").read_text().split("<<'PY'\n", 1)[1].split("\nPY", 1)[0]
    monkeypatch.setenv("EXPECTED_VERSION", "0.4.7")
    class Response:
        headers = {"Content-Type": "application/json"}
        server_version = "0.4.7"
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def read(self):
            return json.dumps({"id": 1, "result": {"serverInfo": {
                "name": "podcast-digest", "version": self.server_version}, "capabilities": {"tools": {}}}}).encode()
    monkeypatch.setattr(urllib.request, "urlopen", lambda *a, **k: Response())
    exec(code, {})
    Response.server_version = "4.0.10"
    with pytest.raises(AssertionError):
        exec(code, {})


def test_publishers_pass_product_versions_to_each_build_and_smoke():
    import yaml
    checked_workflows = 0
    for path in [ROOT / ".github/workflows/release.yml", ROOT / "publish/overlay/.github/workflows/release.yml"]:
        if not path.exists():  # Private repo keeps only the public overlay; public export has .github/workflows.
            continue
        checked_workflows += 1
        workflow = yaml.safe_load(path.read_text())
        steps = workflow["jobs"]["build-and-push"]["steps"]
        builds = [step["with"] for step in steps if step.get("uses", "").startswith("docker/build-push-action")]
        assert len(builds) == 3
        for build in builds:
            args = build["build-args"]
            assert "VCS_REF=${{ github.sha }}" in args
            assert "org.opencontainers.image.version=${{ steps.source.outputs.version }}" in build["labels"]
            assert build["provenance"] == "mode=max" and build["sbom"] is True
            if build["target"] == "transcriber-runtime":
                assert "VERSION=${{ steps.source.outputs.transcriber_version }}" in args
            else:
                assert "VERSION=${{ steps.source.outputs.version }}" in args
            if build["target"] == "mcp-runtime":
                assert "TRANSCRIBER_VERSION=${{ steps.source.outputs.transcriber_version }}" in args
        smokes = [step["run"] for step in steps if "scripts/smoke-image.sh" in step.get("run", "")]
        assert len(smokes) == 3
        assert all('"${{ steps.source.outputs.transcriber_version }}"' in command for command in smokes)
    assert checked_workflows >= 1

    native_release = ROOT / ".github/jenkins/ci-release.sh"
    if native_release.exists():
        script = native_release.read_text(encoding="utf-8")
        assert script.count("--provenance=mode=max") == 3
        assert script.count('--attest "type=sbom,generator=$gen"') == 3
        assert '--build-arg "VERSION=$transcriber_version"' in script
        assert '--build-arg "TRANSCRIBER_IMAGE=$transcriber_repo@$t_digest"' in script
        assert '--build-arg "TRANSCRIBER_VERSION=$transcriber_version"' in script
        for variant in ("transcriber", "podcast", "mcp"):
            assert f'bash scripts/smoke-image.sh {variant} "$image" "$FLEET_SHA" "$version" "$transcriber_version"' in script


def test_rich_sdk_sink_keeps_stream_threshold_and_filters(clean_logging):
    from rich.console import Console
    from rich.logging import RichHandler

    stream = io.StringIO()
    original = RichHandler(console=Console(file=stream, force_terminal=False, width=240),
                           show_time=False, show_level=False, show_path=False)
    original.setLevel(logging.ERROR)
    original.setFormatter(logging.Formatter("sanitized", validate=False))

    def allowed(record):
        return record.msg != "drop"

    original.addFilter(allowed)
    logger = logging.getLogger("mcp.rich_fixture")
    logger.handlers = [original]
    logger.setLevel(logging.ERROR)
    logger.propagate = False
    log.configure("DEBUG", mcp=True)
    handler = logger.handlers[0]
    assert handler.level == logging.ERROR and allowed in handler.filters
    assert handler.stream is stream
    logger.warning("suppressed")
    logger.error("drop")
    logger.error("sensitive content")
    assert len(stream.getvalue().splitlines()) == 1
    payload = json.loads(stream.getvalue())
    assert payload["msg"].rstrip() == "sanitized" and payload["service"] == "podcast-digest-mcp"


def test_transcriber_cli_stdout_remains_artifact_protocol(clean_logging, monkeypatch, tmp_path, capsys):
    from local_transcriber import cli

    manifest = tmp_path / "manifest.json"
    artifact = {"schema_version": 2, "artifact_id": "fictional-episode"}
    manifest.write_text(json.dumps(artifact) + "\n")

    async def fake_transcribe(*args, **kwargs):
        return types.SimpleNamespace(manifest_path=manifest)

    monkeypatch.setattr(cli, "transcribe_file", fake_transcribe)
    assert cli.main(["fictional.mp3", "--output-dir", str(tmp_path),
                     "--artifact-id", "fictional-episode", "--model", "fictional.bin"]) == 0
    assert json.loads(capsys.readouterr().out) == artifact


def test_owned_startup_diagnostic_retains_prefix_and_identity(monkeypatch, tmp_path, capsys):
    from podcast_digest.__main__ import main
    from podcast_digest_mcp.server import build_app

    assert main(["check-config", str(tmp_path / "absent.toml")]) == 1
    output = capsys.readouterr()
    assert output.out == "" and output.err.startswith("config invalid:")
    assert "service=podcast-digest " in output.err and "source_revision=" in output.err
    monkeypatch.delenv("PODCAST_MCP_BEARER_TOKEN", raising=False)
    with pytest.raises(RuntimeError, match="service=podcast-digest-mcp "):
        build_app()


@pytest.mark.parametrize("standalone", [False, True])
def test_redacted_json_identity_collisions_are_preserved(clean_logging, standalone):
    from local_transcriber import log as transcriber_log

    class Redactor(logging.Formatter):
        def format(self, record):
            return json.dumps({"msg": "sanitized", "service": {"original": True},
                               "service_": 7, "service__": False, "version": 18})

    formatter = (transcriber_log if standalone else log)._SinkFormatter(
        Redactor(), name_field="name", tty=False)
    record = logging.LogRecord("local_transcriber.worker" if standalone else "podcast_digest.worker",
                               logging.ERROR, "", 0, "private", (), None)
    payload = json.loads(formatter.format(record))
    assert payload["service"] == ("local-transcriber" if standalone else "podcast-digest")
    assert payload["service_"] == 7 and payload["service__"] is False
    assert payload["service___"] == {"original": True}
    assert payload["version_"] == 18
    assert "private" not in json.dumps(payload)


@pytest.mark.parametrize("module,args,service", [
    ("podcast_digest", ["run", "--credential=synthetic-secret"], "podcast-digest"),
    ("local_transcriber.cli", ["input", "--threads", "synthetic-secret"], "local-transcriber"),
])
def test_argument_errors_are_attributed_without_echoing_values(module, args, service):
    import subprocess
    import sys

    result = subprocess.run([sys.executable, "-m", module, *args], capture_output=True, text=True)
    assert result.returncode == 2 and result.stdout == ""
    assert "synthetic-secret" not in result.stderr
    payload = json.loads(result.stderr)
    assert payload["service"] == service and payload["instance"]
    assert "--help" in payload["msg"]
    help_result = subprocess.run([sys.executable, "-m", module, "--help"], capture_output=True, text=True)
    assert help_result.returncode == 0 and help_result.stderr == ""
    assert help_result.stdout.startswith("usage:") and "source_revision" not in help_result.stdout


@pytest.mark.parametrize("module,args,extra_env,service", [
    ("podcast_digest", ["run", "--dry-run"],
     {"PODCAST_DIGEST_SMTP_PORT": "synthetic-private-value"}, "podcast-digest"),
    ("local_transcriber.cli", ["synthetic-private-file", "--output-dir", "/tmp/unused",
                              "--artifact-id", "fictional", "--model", "fictional.bin"],
     {}, "local-transcriber"),
    ("podcast_digest_mcp", [], {"PODCAST_MCP_BEARER_TOKEN": "", "PORT": "synthetic-private-port"},
     "podcast-digest-mcp"),
])
def test_top_level_failures_keep_exit_one_and_suppress_private_details(module, args, extra_env, service):
    import os
    import subprocess
    import sys

    result = subprocess.run([sys.executable, "-m", module, *args], capture_output=True,
                            text=True, env={**os.environ, **extra_env})
    assert result.returncode == 1 and result.stdout == ""
    assert "synthetic-private" not in result.stderr and "Traceback" not in result.stderr
    payload = json.loads(result.stderr)
    assert payload["service"] == service and payload["instance"]
    assert "details withheld" in payload["msg"]


def test_startup_reasons_remain_actionable_without_values_or_paths(tmp_path):
    import os
    import subprocess
    import sys

    environment = {**os.environ, "PODCAST_DIGEST_DATA_DIR": str(tmp_path),
                   "PODCAST_DIGEST_SUMMARIZER": "api", "PODCAST_DIGEST_MODEL": "claude-opus-5-5",
                   "ANTHROPIC_API_KEY": "", "OPENAI_API_KEY": "",
                   "PODCAST_DIGEST_OPENAI_BASE_URL": ""}
    worker = subprocess.run([sys.executable, "-m", "podcast_digest", "run", "--dry-run"],
                            env=environment, capture_output=True, text=True, timeout=15)
    payload = json.loads(worker.stderr)
    assert worker.returncode == 1 and worker.stdout == ""
    assert payload["reason"] == "summarizer_configuration"
    assert "ANTHROPIC_API_KEY" in payload["settings"]
    for overrides, setting in [
        ({"PODCAST_DIGEST_SUMMARIZER": "command", "PODCAST_DIGEST_COMMAND": ""},
         "PODCAST_DIGEST_COMMAND"),
        ({"PODCAST_DIGEST_WEBHOOK_FIELDS": "synthetic-private-json"},
         "PODCAST_DIGEST_WEBHOOK_FIELDS"),
    ]:
        invalid = subprocess.run([sys.executable, "-m", "podcast_digest", "run", "--dry-run"],
                                 env={**environment, **overrides}, capture_output=True, text=True, timeout=15)
        diagnostic = json.loads(invalid.stderr)
        assert invalid.returncode == 1 and invalid.stdout == ""
        assert diagnostic["reason"] == "invalid_settings" and setting in diagnostic["settings"]
        assert "synthetic-private-json" not in invalid.stderr and "Traceback" not in invalid.stderr
    for overrides, reason, setting in [
        ({"PODCAST_MCP_BEARER_TOKEN": ""}, "missing_bearer_token", "PODCAST_MCP_BEARER_TOKEN"),
        ({"PODCAST_MCP_BEARER_TOKEN": "synthetic-only", "PORT": "private-invalid-port"},
         "invalid_listener_port", "PORT"),
    ]:
        result = subprocess.run([sys.executable, "-m", "podcast_digest_mcp"],
                                env={**environment, **overrides}, capture_output=True, text=True, timeout=15)
        diagnostic = json.loads(result.stderr)
        assert result.returncode == 1 and result.stdout == ""
        assert diagnostic["reason"] == reason and diagnostic["setting"] == setting
        assert "private-invalid-port" not in result.stderr and "synthetic-only" not in result.stderr
    transcriber = subprocess.run([
        sys.executable, "-m", "local_transcriber.cli", "private-missing-media", "--model", "private-model",
        "--output-dir", str(tmp_path), "--artifact-id", "fictional",
    ], env=environment, capture_output=True, text=True, timeout=15)
    assert transcriber.returncode == 1 and transcriber.stdout == ""
    assert "reason=input_file_missing" in json.loads(transcriber.stderr)["msg"]
    assert "private-missing-media" not in transcriber.stderr and "private-model" not in transcriber.stderr


def test_mcp_http_access_and_startup_keep_their_streams(tmp_path):
    program = """import logging
import uvicorn
from podcast_digest_mcp import server
def run(app, **kwargs):
    logging.getLogger('uvicorn.access').info('%s - "%s %s HTTP/%s" %d',
        '127.0.0.1:12345', 'GET', '/health', '1.1', 200)
    logging.getLogger('uvicorn.error').error('synthetic startup error')
uvicorn.run = run
server.main()
"""
    result = subprocess.run(
        [sys.executable, "-c", program], capture_output=True, text=True, timeout=30, check=True,
        env={**os.environ, "PODCAST_MCP_BEARER_TOKEN": "synthetic-test-token", "LOG_LEVEL": "INFO",
             "PODCAST_DIGEST_CONFIG": str(tmp_path / "feeds.toml")},
    )
    access, = [json.loads(line) for line in result.stdout.splitlines()]
    assert access["logger"] == "uvicorn.access" and access["lvl"] == "INFO"
    assert '\"GET /health HTTP/1.1\" 200 OK' in access["msg"]
    assert access["service"] == "podcast-digest-mcp" and access["version"] == PODCAST_VERSION
    errors = [json.loads(line) for line in result.stderr.splitlines()]
    assert all(item["logger"] != "uvicorn.access" for item in errors)
    startup = next(item for item in errors if item["logger"] == "uvicorn.error")
    assert startup["msg"] == "synthetic startup error" and startup["lvl"] == "ERROR"
    assert startup["service"] == access["service"] and startup["instance"] == access["instance"]
