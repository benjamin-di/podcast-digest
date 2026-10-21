from __future__ import annotations

from podcast_digest.__main__ import main


def test_check_config_validates_without_runtime_credentials(tmp_path, capsys):
    path = tmp_path / "feeds.toml"
    path.write_text(
        'recipient = "reader@example.com"\n'
        "poll_seconds = 60\n"
        "min_minutes = 5\n"
        "[[feeds]]\n"
        'name = "Example"\n'
        'rss_url = "https://example.com/feed.xml"\n',
        encoding="utf-8",
    )
    assert main(["check-config", str(path)]) == 0

    output = capsys.readouterr()
    assert output.err == ""
    assert f"config ok: {path}" in output.out
    assert "feeds=1" in output.out
    assert "rss_url=1" in output.out
    assert list(tmp_path.iterdir()) == [path]


def test_check_config_reports_schema_error_without_traceback(tmp_path, capsys):
    path = tmp_path / "feeds.toml"
    path.write_text(
        'recipient = "reader@example.com"\n'
        "[[feeds]]\n"
        'name = "Missing source"\n',
        encoding="utf-8",
    )

    assert main(["check-config", str(path)]) == 1

    output = capsys.readouterr()
    assert output.out == ""
    assert output.err.startswith("config invalid:")
    assert "needs exactly one source" in output.err


def test_check_config_defaults_to_repository_file(monkeypatch, tmp_path, capsys):
    (tmp_path / "feeds.toml").write_text(
        'recipient = "reader@example.com"\nfeeds = []\n',
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("PODCAST_DIGEST_CONFIG", raising=False)

    assert main(["check-config"]) == 0
    assert "config ok: feeds.toml | feeds=0" in capsys.readouterr().out


def test_version_flag(capsys):
    import pytest

    from podcast_digest import __version__
    from podcast_digest.__main__ import main

    with pytest.raises(SystemExit) as ei:
        main(["--version"])
    assert ei.value.code == 0
    assert capsys.readouterr().out.strip() == f"podcast-digest {__version__}"
