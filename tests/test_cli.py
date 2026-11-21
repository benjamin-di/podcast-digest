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


def test_delivery_inspection_and_reconciliation_need_no_summarizer(
    settings, db, monkeypatch, capsys, tmp_path,
):
    import json

    from podcast_digest.mail import GraphMailer

    from .test_delivery import seed

    envelope = seed(db)
    log_path = tmp_path / "emitted.tsv"
    monkeypatch.setenv("PODCAST_DIGEST_EMITTED_LOG", str(log_path))
    audio = tmp_path / "synthetic.mp3"
    audio.write_bytes(b"synthetic audio")
    db.set_status(feed_url="https://example.com/feed.xml", guid="synthetic-digest",
                  status="summarized", audio_path=str(audio))
    graph = GraphMailer(tenant_id="test", client_id="test", client_secret="test",
                        sender="bot@example.com")
    row = db.claim_delivery("https://example.com/feed.xml", "synthetic-digest",
                            target=graph.delivery_target, envelope=envelope.serialize())
    monkeypatch.setattr("podcast_digest.__main__.build_mailer", lambda _settings: graph)
    monkeypatch.setattr(graph, "sent_receipt", lambda **_kwargs: None)
    assert main(["delivery", "list"]) == 0
    assert f"{row.operation_id} unknown attempts=1" in capsys.readouterr().out
    assert main(["delivery", "inspect", row.operation_id]) == 0
    inspected = json.loads(capsys.readouterr().out)
    assert inspected["envelope"]["body_html"] == envelope.body_html
    assert inspected["receipt"] is None
    command = ["delivery", "reconcile", row.operation_id, "--graph-message-id", "synthetic"]
    assert main(command) == 1
    assert "held" in capsys.readouterr().err
    assert db.delivery_by_id(row.operation_id).state == "unknown"
    monkeypatch.setattr(graph, "sent_receipt", lambda **_kwargs: "graph-sent-item:synthetic")
    assert main(command) == 0
    assert "accepted" in capsys.readouterr().out
    assert db.get_episode(row.feed_url, row.guid).status == "emailed"
    from podcast_digest.emitted import EmittedLog

    assert EmittedLog.load(log_path).contains(row.feed_url, row.guid)
    assert db.delivery_by_id(row.operation_id).finalized_at is not None
    assert not audio.exists()
    assert main(command) == 0
    assert len(log_path.read_text().splitlines()) == 1


def test_delivery_recovery_obeys_tick_lock(settings, monkeypatch, capsys):
    from podcast_digest.pipeline import _acquire_tick_lock

    lock = _acquire_tick_lock(settings)
    assert lock is not None
    try:
        assert main(["delivery", "list"]) == 1
        assert "another tick or recovery command" in capsys.readouterr().err
    finally:
        lock.close()
