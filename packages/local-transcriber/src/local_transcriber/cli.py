"""Command-line entry point for local transcription workers."""

from __future__ import annotations

import argparse
import asyncio
import os
from pathlib import Path

from .models import WhisperCppConfig
from .whisper import transcribe_file


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="local-transcribe")
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
    args = _parser().parse_args(argv)
    return asyncio.run(_run(args))


if __name__ == "__main__":
    raise SystemExit(main())
