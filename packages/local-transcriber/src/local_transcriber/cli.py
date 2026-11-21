"""Command-line entry point for local transcription workers."""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
from pathlib import Path
from typing import Never

from .log import configure
from .models import WhisperCppConfig
from .whisper import transcribe_file


class _ArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> Never:
        # Argument values can contain credentials or private file names.
        logging.getLogger(__name__).error("Invalid command arguments; use --help for usage.")
        raise SystemExit(2)


def _parser() -> argparse.ArgumentParser:
    parser = _ArgumentParser(prog="local-transcribe")
    parser.add_argument("input", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--artifact-id", required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--whisper-bin", default="whisper-cli")
    parser.add_argument("--language", default="en")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--device", default="GPU")
    parser.add_argument("--library-path")
    parser.add_argument("--initial-prompt")
    parser.add_argument("--engine-version", default=os.environ.get("WHISPER_CPP_VERSION", "unknown"))
    parser.add_argument("--lock-path", type=Path)
    return parser


async def _run(args: argparse.Namespace) -> int:
    result = await transcribe_file(
        args.input,
        output_dir=args.output_dir,
        artifact_id=args.artifact_id,
        config=WhisperCppConfig(
            whisper_bin=args.whisper_bin,
            model_path=args.model,
            language=args.language,
            threads=args.threads,
            device=args.device or None,
            library_path=args.library_path,
            initial_prompt=args.initial_prompt,
            engine_version=args.engine_version,
            lock_path=args.lock_path,
        ),
    )
    print(result.manifest_path.read_text(encoding="utf-8").strip())
    return 0


def main(argv: list[str] | None = None) -> int:
    configure()
    args = _parser().parse_args(argv)
    try:
        return asyncio.run(_run(args))
    except Exception as exc:
        # Child stderr and private paths are not a diagnostic protocol.
        reasons = {
            "media not found:": "input_file_missing",
            "whisper model not found:": "model_file_missing",
            "ffmpeg binary not found:": "ffmpeg_binary_missing",
            "whisper binary not found:": "whisper_binary_missing",
            "artifact_id must be": "invalid_artifact_id",
            "ffmpeg failed": "ffmpeg_process_failed",
            "whisper-cli failed": "whisper_process_failed",
            "whisper JSON output missing:": "whisper_output_missing",
            "invalid whisper JSON": "invalid_whisper_output",
            "whisper produced no text:": "empty_whisper_output",
        }
        reason = next((value for prefix, value in reasons.items() if str(exc).startswith(prefix)),
                      "transcription_failed")
        logging.getLogger(__name__).error(
            "Transcription command failed; details withheld. reason=%s error_type=%s",
            reason, type(exc).__name__)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
