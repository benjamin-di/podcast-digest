"""Fresh-process first use, concurrent readers, and a real fork boundary."""

import subprocess
import sys

import pytest

PROBE = r"""
import importlib
import json
import logging
import os
import re
import sys
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

original_uuid4 = uuid.uuid4


def slow_uuid4():
    time.sleep(0.03)
    return original_uuid4()


# Make simultaneous first calls overlap even on a fast local machine.
uuid.uuid4 = slow_uuid4
module = importlib.import_module(sys.argv[1])
accessor = sys.argv[2]


def instance():
    if accessor == "formatter":
        record = logging.LogRecord(
            "synthetic", logging.WARNING, "", 0, "retained", (), None
        )
        return json.loads(module.JsonFormatter("synthetic").format(record))["instance"]
    value = getattr(module, accessor)()
    return value if isinstance(value, str) else value["instance"]


def concurrent_instances():
    barrier = Barrier(8)

    def read(_):
        barrier.wait(timeout=5)
        return instance()

    with ThreadPoolExecutor(max_workers=8) as pool:
        values = set(pool.map(read, range(8)))
    assert len(values) == 1, values
    value = values.pop()
    assert re.fullmatch(r"[a-f0-9]{32}", value), value
    assert instance() == value
    return value


parent = concurrent_instances()
if hasattr(os, "fork"):
    reader, writer = os.pipe()
    child = os.fork()
    if child == 0:
        os.close(reader)
        try:
            value = concurrent_instances()
            os.write(writer, value.encode("ascii"))
        except BaseException:
            os._exit(1)
        finally:
            os.close(writer)
        os._exit(0)
    os.close(writer)
    with os.fdopen(reader, "rb") as pipe:
        value = pipe.read().decode("ascii")
    _, status = os.waitpid(child, 0)
    assert os.waitstatus_to_exitcode(status) == 0
    assert re.fullmatch(r"[a-f0-9]{32}", value), value
    assert value != parent, (value, parent)
    assert instance() == parent
"""


@pytest.mark.parametrize(
    "module,accessor",
    [("podcast_digest.identity", "log_fields"), ("local_transcriber.identity", "log_fields")],
)
def test_process_instance_is_stable_for_threads_and_fork(module, accessor):
    result = subprocess.run(
        [sys.executable, "-c", PROBE, module, accessor],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
