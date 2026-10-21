from __future__ import annotations

import pytest
from pydantic import ValidationError

from podcast_digest.config import Feed, FeedsFile, Settings
from podcast_digest.llm import ApiCompleter, build_completer
from podcast_digest.mail import SmtpMailer, build_mailer


def _settings(**overrides) -> Settings:
    return Settings(**overrides)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("PODCAST_DIGEST_SUMMARIZER", "remote"),
        ("PODCAST_DIGEST_PROVIDER", "gemini"),
        ("PODCAST_DIGEST_COMMAND_PROTOCOL", "xml"),
        ("PODCAST_DIGEST_SUMMARY_STYLE", "poetry"),
        ("PODCAST_DIGEST_ATTACH_TRANSCRIPT", "markdown"),
        ("PODCAST_DIGEST_MAILER", "sendgrid"),
        ("PODCAST_DIGEST_SMTP_SECURITY", "tls"),
        ("PODCAST_DIGEST_SMTP_PORT", 0),
        ("PODCAST_DIGEST_WHISPER_THREADS", 0),
        ("PODCAST_DIGEST_WAKE_POLL_SECONDS", 0),
        ("PODCAST_DIGEST_SSH_PORT", 0),
        ("PODCAST_DIGEST_SSH_PORT", 65536),
        ("PODCAST_DIGEST_COMMAND_TIMEOUT", 0),
        ("PODCAST_DIGEST_WEBHOOK_TIMEOUT", 0),
    ],
)
def test_settings_reject_invalid_modes_and_ranges(field, value):
    with pytest.raises(ValidationError):
        _settings(**{field: value})


def test_defaults_need_no_credentials_and_use_the_api_backend():
    s = _settings()
    assert s.summarizer_backend == "api"
    assert s.mailer_backend == "smtp"
    assert s.PODCAST_DIGEST_SUMMARY_STYLE == "general"
    assert isinstance(build_completer(s), ApiCompleter)


def test_command_mode_requires_a_command():
    with pytest.raises(ValidationError, match="PODCAST_DIGEST_COMMAND is required"):
        _settings(PODCAST_DIGEST_SUMMARIZER="command")


def test_command_protocol_defaults_to_text_and_runs_locally():
    s = _settings(PODCAST_DIGEST_SUMMARIZER="command", PODCAST_DIGEST_COMMAND="llm -m x")
    assert s.summarizer_backend == "command"
    assert s.command_protocol == "text"
    completer = build_completer(s)
    assert completer.label == "command:text"


def test_cleaned_attachment_works_with_any_backend():
    s = _settings(PODCAST_DIGEST_ATTACH_TRANSCRIPT="cleaned")
    assert s.PODCAST_DIGEST_ATTACH_TRANSCRIPT == "cleaned"


def test_command_model_only_when_explicit(monkeypatch):
    assert _settings().command_model is None
    monkeypatch.setenv("PODCAST_DIGEST_MODEL", "claude-sonnet-5")
    assert _settings().command_model == "claude-sonnet-5"



def test_webhook_fields_parse_from_json_env(monkeypatch):
    monkeypatch.setenv("PODCAST_DIGEST_WEBHOOK_FIELDS", '{"source": "podcasts"}')
    assert _settings().webhook_fields == {"source": "podcasts"}


def test_build_mailer_smtp_and_its_requirements():
    with pytest.raises(ValueError, match="MAIL_FROM"):
        build_mailer(_settings(PODCAST_DIGEST_SMTP_HOST="smtp.example.com"))
    with pytest.raises(ValueError, match="SMTP_HOST"):
        build_mailer(_settings(PODCAST_DIGEST_MAIL_FROM="bot@example.com"))
    mailer = build_mailer(_settings(
        PODCAST_DIGEST_MAIL_FROM="bot@example.com",
        PODCAST_DIGEST_SMTP_HOST="smtp.example.com",
        PODCAST_DIGEST_SMTP_SECURITY="ssl",
        PODCAST_DIGEST_SMTP_PORT=465,
    ))
    assert isinstance(mailer, SmtpMailer)
    assert (mailer.host, mailer.port, mailer.security) == ("smtp.example.com", 465, "ssl")


def test_explicit_graph_mailer_names_missing_credentials():
    with pytest.raises(ValueError, match="GRAPH_CLIENT_SECRET"):
        build_mailer(_settings(
            PODCAST_DIGEST_MAILER="graph",
            PODCAST_DIGEST_MAIL_FROM="bot@example.com",
            PODCAST_DIGEST_GRAPH_TENANT_ID="t",
            PODCAST_DIGEST_GRAPH_CLIENT_ID="c",
        ))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"poll_seconds": 0},
        {"poll_seconds": -1},
        {"min_minutes": -1},
    ],
)
def test_feeds_file_rejects_invalid_ranges(kwargs):
    with pytest.raises(ValidationError):
        FeedsFile(
            recipient="reader@example.com",
            feeds=[Feed(name="Valid", search="valid")],
            **kwargs,
        )


def test_feeds_file_rejects_blank_recipient():
    with pytest.raises(ValidationError, match="recipient"):
        FeedsFile(recipient="  ", feeds=[Feed(name="Valid", search="valid")])


def test_valid_zero_minimum():
    feeds = FeedsFile(
        recipient="reader@example.com",
        min_minutes=0,
        feeds=[Feed(name="Clips", youtube="@clips", min_minutes=0)],
    )
    assert feeds.min_minutes_for(feeds.feeds[0]) == 0





def test_webhook_fields_accept_non_string_values(monkeypatch):
    monkeypatch.setenv("PODCAST_DIGEST_WEBHOOK_FIELDS", '{"priority": 2, "tags": ["a"]}')
    assert _settings().webhook_fields == {"priority": 2, "tags": ["a"]}



def test_empty_variables_count_as_unset(monkeypatch):
    """Compose wiring like `X: ${Y:-}` yields empty strings: they must fall back
    to the default instead of breaking a typed setting."""
    for name in ("PODCAST_DIGEST_WEBHOOK_FIELDS", "PODCAST_DIGEST_MAILER",
                 "PODCAST_DIGEST_COMMAND_PROTOCOL", "PODCAST_DIGEST_SSH_PORT",
                 "PODCAST_DIGEST_SUMMARIZER"):
        monkeypatch.setenv(name, "")
    s = _settings()
    assert s.webhook_fields == {}
    assert s.PODCAST_DIGEST_SSH_PORT == 22
    assert s.summarizer_backend == "api"


def test_settings_ignore_a_root_env_file(tmp_path, monkeypatch):
    """The quickstart's .env in the repository root must not leak into tests."""
    (tmp_path / ".env").write_text("PODCAST_DIGEST_SUMMARIZER=command\n")
    monkeypatch.chdir(tmp_path)
    assert _settings().summarizer_backend == "api"


def test_webhook_ack_field_and_feed_opt_out():
    s = _settings(PODCAST_DIGEST_WEBHOOK_URL="https://hooks.example.com/p",
                  PODCAST_DIGEST_WEBHOOK_ACK_FIELD=" task_id ")
    assert (s.webhook_url, s.webhook_ack_field) == ("https://hooks.example.com/p", "task_id")
    assert Feed.model_validate({"name": "A", "search": "a", "webhook": False}).webhook is False
    assert Feed(name="A", search="a").webhook is True
