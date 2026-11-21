"""Keep host capture and native Rich error delivery intact during attribution."""

import io
import logging
import threading
import unittest

try:
    from rich.console import Console
    from rich.logging import RichHandler
except ImportError as error:
    raise unittest.SkipTest("Optional Rich delivery tests require Rich") from error

from podcast_digest import log as logs


class DeliveryFailure(Exception):
    pass


class NativeHandler(RichHandler):
    def __init__(self, console):
        super().__init__(
            console=console,
            show_time=False,
            show_level=False,
            show_path=False,
            highlighter=None,
        )
        self.errors = []
        self.callback_error = None

    def handleError(self, record):
        self.errors.append(record)
        if self.callback_error is not None:
            raise self.callback_error


class BrokenStream(io.StringIO):
    def __init__(self, operation):
        super().__init__()
        self.operation = operation

    def write(self, value):
        if self.operation == "write":
            raise OSError("synthetic write failure")
        return super().write(value)

    def flush(self):
        if self.operation == "flush":
            raise OSError("synthetic flush failure")
        return super().flush()


def record():
    return logging.LogRecord(
        "synthetic.delivery", logging.INFO, __file__, 1, "retained diagnostic", (), None
    )


def adapter(original):
    wrapped = logs._RichSink(
        original, logs._SinkFormatter(logging.Formatter(), name_field="name", tty=False)
    )
    wrapped.identity_formatter = logging.Formatter("attributed %(message)s")
    wrapped_errors = []
    wrapped.handleError = wrapped_errors.append
    return wrapped, wrapped_errors


def actual_adapter(original, tty=False):
    wrapped = logs._RichSink(
        original, logs._SinkFormatter(logging.Formatter(), name_field="name", tty=False)
    )
    if hasattr(wrapped.identity_formatter, "tty"):
        wrapped.identity_formatter.tty = tty
    wrapped_errors = []
    wrapped.handleError = wrapped_errors.append
    return wrapped, wrapped_errors


class RichDeliveryContractTests(unittest.TestCase):
    def setUp(self):
        from unittest.mock import patch

        fields = {
            "service": "synthetic-service",
            "version": "1.2.3",
            "source_revision": "a" * 40,
            "instance": "synthetic-instance",
        }
        for name in ("log_fields", "log_context"):
            if hasattr(logs, name):
                identity = patch.object(logs, name, return_value=fields)
                self.addCleanup(identity.stop)
                identity.start()

    def assert_released(self, *handlers):
        for handler in handlers:
            lock_results = []

            def acquire_from_another_thread(handler=handler, lock_results=lock_results):
                locked = handler.lock.acquire(timeout=1)
                lock_results.append(locked)
                if locked:
                    handler.lock.release()

            worker = threading.Thread(target=acquire_from_another_thread)
            worker.start()
            worker.join(timeout=2)
            self.assertFalse(worker.is_alive())
            self.assertEqual(lock_results, [True])

    def test_native_console_state_reads_writes_and_deletes_are_preserved(self):
        sink = io.StringIO()
        console = Console(file=sink, record=False, force_terminal=False)
        console.synthetic_policy = "retained"
        seen = []

        class StatefulHandler(NativeHandler):
            def emit(self, effective):
                seen.append(self.console.record)
                self.console.quiet = True
                del self.console.synthetic_policy
                super().emit(effective)

        original = StatefulHandler(console)
        wrapped, errors = adapter(original)
        wrapped.handle(record())
        self.assertEqual(seen, [False])
        self.assertTrue(console.quiet)
        self.assertFalse(hasattr(console, "synthetic_policy"))
        self.assertEqual(sink.getvalue(), "")
        self.assertEqual(original.errors, [])
        self.assertEqual(errors, [])
        self.assertIs(original.console, console)

    def test_native_error_recovery_can_replace_console_file(self):
        initial, fallback = io.StringIO(), io.StringIO()
        console = Console(file=initial, force_terminal=False)

        class RecoveringHandler(NativeHandler):
            def emit(self, effective):
                self.handleError(effective)

            def handleError(self, effective):
                self.errors.append(effective)
                self.console.file = fallback
                self.console.print("recovered diagnostic", markup=False, highlight=False)

        original = RecoveringHandler(console)
        wrapped, errors = adapter(original)
        wrapped.handle(record())
        self.assertEqual(initial.getvalue(), "")
        self.assertIn("recovered diagnostic", fallback.getvalue())
        self.assertIs(console.file, fallback)
        self.assertEqual(len(original.errors), 1)
        self.assertEqual(errors, [])
        self.assertIs(original.console, console)

    def test_effective_instance_print_redaction_and_suppression_are_preserved(self):
        from functools import partial
        from types import MethodType

        for callable_kind in ("bound", "closure", "partial"):
            for suppress in (False, True):
                with self.subTest(callable_kind=callable_kind, suppress=suppress):
                    sink = io.StringIO()
                    console = Console(file=sink, force_terminal=False)
                    native_print = console.print
                    calls = []

                    def redact(
                        *objects,
                        calls=calls,
                        suppress=suppress,
                        native_print=native_print,
                        **kwargs,
                    ):
                        calls.append(objects)
                        if not suppress:
                            native_print("redacted diagnostic", **kwargs)

                    def bound_redact(receiver, *objects, console=console, redact=redact, **kwargs):
                        self.assertIs(receiver, console)
                        return redact(*objects, **kwargs)

                    console.print = (
                        MethodType(bound_redact, console)
                        if callable_kind == "bound"
                        else partial(redact)
                        if callable_kind == "partial"
                        else redact
                    )
                    effective_print = console.print
                    original = NativeHandler(console)
                    wrapped, errors = adapter(original)
                    hooks = console._render_hooks
                    with console.capture() as captured:
                        wrapped.handle(record())
                    output = captured.get()
                    self.assertEqual(len(calls), 1)
                    self.assertNotIn("retained diagnostic", output)
                    if suppress:
                        self.assertEqual(output, "")
                    else:
                        self.assertIn("redacted diagnostic", output)
                        self.assertGreater(len(output), len("redacted diagnostic\n"))
                    self.assertEqual(sink.getvalue(), "")
                    self.assertEqual(original.errors, [])
                    self.assertEqual(errors, [])
                    self.assertIs(console.print, effective_print)
                    self.assertIs(console._render_hooks, hooks)
                    self.assertNotIn("render", vars(console))
                    self.assertNotIn("_build_identity_render_dispatch", vars(console))

    def test_effective_unbound_print_keeps_live_host_output_unattributed(self):
        from rich.live import Live

        sink = io.StringIO()
        console = Console(
            file=sink,
            width=80,
            force_terminal=True,
            force_interactive=True,
            color_system="standard",
            _environ={"TERM": "xterm"},
        )
        native_print = console.print

        def effective_print(*objects, **kwargs):
            return native_print(*objects, **kwargs)

        console.print = effective_print
        original = NativeHandler(console)
        wrapped, errors = adapter(original)
        hooks = console._render_hooks
        live = Live(
            "host status",
            console=console,
            auto_refresh=False,
            redirect_stdout=False,
            redirect_stderr=False,
        )
        live.start(refresh=True)
        try:
            sink.seek(0)
            sink.truncate()
            wrapped.handle(record())
            emitted = sink.getvalue()
        finally:
            live.stop()
        self.assertTrue(emitted.startswith("\r\x1b[2K"), repr(emitted))
        self.assertTrue(emitted.endswith("host status"), repr(emitted))
        self.assertEqual(emitted.count("host status"), 1)
        self.assertIn("retained diagnostic", emitted)
        self.assertGreater(len(emitted.split("retained diagnostic")[0]), len("\r\x1b[2K"))
        self.assertEqual(original.errors, [])
        self.assertEqual(errors, [])
        self.assertIs(console._render_hooks, hooks)
        self.assertNotIn("render", vars(console))

    def test_two_handlers_share_console_without_cross_thread_attribution(self):
        sink = io.StringIO()
        console = Console(file=sink, force_terminal=False, width=80)
        native_print = console.print
        first_entered, second_entered = threading.Event(), threading.Event()
        outputs, failures = {}, []

        def effective_print(*objects, **kwargs):
            name = threading.current_thread().name
            if name == "first-record":
                first_entered.set()
                if not second_entered.wait(2):
                    raise DeliveryFailure("second print could not enter")
            elif name == "second-record":
                second_entered.set()
            return native_print(*objects, **kwargs)

        console.print = effective_print
        originals = [NativeHandler(console), NativeHandler(console)]
        wrapped = [adapter(original)[0] for original in originals]

        def emit(index):
            try:
                with console.capture() as captured:
                    native_print("host prefix", highlight=False)
                    wrapped[index].handle(record())
                    native_print("host suffix", highlight=False)
                outputs[index] = captured.get()
            except (AssertionError, DeliveryFailure, OSError, RuntimeError) as error:
                failures.append(error)

        workers = [
            threading.Thread(target=emit, args=(index,), name=name)
            for index, name in enumerate(("first-record", "second-record"))
        ]
        workers[0].start()
        try:
            self.assertTrue(first_entered.wait(2))
            workers[1].start()
        finally:
            for worker in workers:
                if worker.ident is not None:
                    worker.join(3)
        self.assertFalse(any(worker.is_alive() for worker in workers))
        self.assertEqual(failures, [])
        self.assertEqual(set(outputs), {0, 1})
        for output in outputs.values():
            self.assertTrue(output.startswith("host prefix\n"), repr(output))
            self.assertTrue(output.endswith("host suffix\n"), repr(output))
            self.assertEqual(output.count("retained diagnostic"), 1)
            self.assertGreater(len(output), len("host prefix\nretained diagnostic\nhost suffix\n"))
        self.assertEqual([original.errors for original in originals], [[], []])
        self.assertNotIn("render", vars(console))
        self.assertNotIn("_build_identity_render_dispatch", vars(console))

    def test_print_override_can_install_a_live_hook_before_native_print(self):
        from rich.live import Live

        class LateLiveConsole(Console):
            def print(self, *objects, **kwargs):
                live = Live(
                    "host status",
                    console=self,
                    auto_refresh=False,
                    redirect_stdout=False,
                    redirect_stderr=False,
                )
                self.push_render_hook(live)
                try:
                    return super().print(*objects, **kwargs)
                finally:
                    self.pop_render_hook()

        sink = io.StringIO()
        console = LateLiveConsole(
            file=sink,
            width=80,
            force_terminal=True,
            force_interactive=True,
            color_system="standard",
            _environ={"TERM": "xterm"},
            record=True,
        )
        hooks = console._render_hooks
        original = NativeHandler(console)
        wrapped, errors = adapter(original)
        wrapped.handle(record())
        self.assertEqual(sink.getvalue().splitlines()[-1], "host status")
        self.assertGreater(len(sink.getvalue().split("retained diagnostic")[0]), len("\r\x1b[2K"))
        self.assertEqual(console.export_text().splitlines()[-1], "host status")
        self.assertIn("retained diagnostic", sink.getvalue())
        self.assertEqual(original.errors, [])
        self.assertEqual(errors, [])
        self.assertIs(console._render_hooks, hooks)
        self.assertEqual(len(hooks), 0)
        self.assertNotIn("render", vars(console))
        self.assertNotIn("_build_identity_render_dispatch", vars(console))

    def test_concurrent_live_start_during_native_print_keeps_host_output(self):
        from rich.live import Live

        entered, resume = threading.Event(), threading.Event()
        failures = []

        class PausedConsole(Console):
            def print(self, *objects, **kwargs):
                if threading.current_thread().name == "record-writer":
                    entered.set()
                    if not resume.wait(3):
                        raise DeliveryFailure("record print timed out")
                return super().print(*objects, **kwargs)

        sink = io.StringIO()
        console = PausedConsole(
            file=sink,
            width=80,
            force_terminal=True,
            force_interactive=True,
            color_system="standard",
            _environ={"TERM": "xterm"},
            record=True,
        )
        hooks = console._render_hooks
        original = NativeHandler(console)
        wrapped, errors = adapter(original)

        def emit():
            try:
                wrapped.handle(record())
            except (AssertionError, DeliveryFailure, OSError, RuntimeError) as error:
                failures.append(error)

        worker = threading.Thread(target=emit, name="record-writer")
        worker.start()
        live = Live(
            "host status",
            console=console,
            auto_refresh=False,
            redirect_stdout=False,
            redirect_stderr=False,
        )
        try:
            self.assertTrue(entered.wait(3))
            live.start(refresh=True)
            sink.seek(0)
            sink.truncate()
            console.export_text()
            resume.set()
            worker.join(4)
            emitted = sink.getvalue()
            recorded = console.export_text()
        finally:
            resume.set()
            worker.join(4)
            live.stop()
        self.assertFalse(worker.is_alive())
        self.assertEqual(failures, [])
        self.assertTrue(emitted.startswith("\r\x1b[2K"), repr(emitted))
        self.assertEqual(emitted.splitlines()[-1], "host status")
        self.assertEqual(recorded.splitlines()[-1], "host status")
        self.assertIn("retained diagnostic", emitted)
        self.assertEqual(original.errors, [])
        self.assertEqual(errors, [])
        self.assertIs(console._render_hooks, hooks)
        self.assertEqual(len(hooks), 0)
        self.assertNotIn("render", vars(console))
        self.assertNotIn("_build_identity_render_dispatch", vars(console))

    def test_host_capture_retains_only_native_record_attribution(self):
        for tty in (False, True):
            with self.subTest(tty=tty):
                sink = io.StringIO()
                console = Console(file=sink, width=80, force_terminal=tty, color_system=None)
                original = NativeHandler(console)
                with console.capture() as native_capture:
                    original.handle(record())
                native = native_capture.get().rstrip("\n")
                wrapped, wrapper_errors = adapter(original)
                with console.capture() as captured:
                    console.out("host prefix", highlight=False)
                    wrapped.handle(record())
                    console.out("host suffix", highlight=False)
                self.assertEqual(
                    captured.get(), "host prefix\nattributed " + native + "\nhost suffix\n"
                )
                self.assertEqual(sink.getvalue(), "")
                self.assertIs(original.console, console)
                self.assertEqual(original.errors, [])
                self.assertEqual(wrapper_errors, [])
                self.assert_released(original, wrapped)

    def test_native_error_callback_owns_write_and_flush_failures(self):
        for tty in (False, True):
            for operation in ("write", "flush"):
                with self.subTest(tty=tty, operation=operation):
                    console = Console(
                        file=BrokenStream(operation), force_terminal=tty, color_system=None
                    )
                    original = NativeHandler(console)
                    wrapped, wrapper_errors = adapter(original)
                    wrapped.handle(record())
                    self.assertEqual(len(original.errors), 1)
                    self.assertEqual(original.errors[0].getMessage(), "retained diagnostic")
                    self.assertEqual(wrapper_errors, [])
                    self.assertIs(original.console, console)
                    self.assert_released(original, wrapped)

    def test_native_formatter_and_error_callback_exceptions_escape_once(self):
        for tty in (False, True):
            for source in ("formatter", "callback"):
                with self.subTest(tty=tty, source=source):
                    sink = BrokenStream("write") if source == "callback" else io.StringIO()
                    console = Console(file=sink, force_terminal=tty, color_system=None)
                    original = NativeHandler(console)
                    wrapped, wrapper_errors = adapter(original)
                    failure = DeliveryFailure("synthetic " + source + " failure")
                    format_calls = []

                    class FailingFormatter(logging.Formatter):
                        def format(self, effective_record, calls=format_calls, error=failure):
                            calls.append(effective_record)
                            raise error

                    if source == "formatter":
                        original.setFormatter(FailingFormatter())
                    else:
                        original.callback_error = failure
                    with self.assertRaises(DeliveryFailure) as raised:
                        wrapped.handle(record())
                    self.assertIs(raised.exception, failure)
                    self.assertEqual(len(format_calls), 1 if source == "formatter" else 0)
                    self.assertEqual(len(original.errors), 1 if source == "callback" else 0)
                    self.assertEqual(wrapper_errors, [])
                    self.assertIs(original.console, console)
                    self.assert_released(original, wrapped)

    def test_subclass_print_preserves_all_objects_and_keyword_options(self):
        from rich.text import Text

        class CustomPrintHandler(NativeHandler):
            def emit(self, effective):
                message = self.format(effective)
                self.console.print(
                    message,
                    Text("second", style="bold red"),
                    sep="|",
                    end="!\n",
                    soft_wrap=True,
                    markup=False,
                    highlight=False,
                )

        for tty in (False, True):
            with self.subTest(tty=tty):
                sink = io.StringIO()
                console = Console(
                    file=sink,
                    width=12,
                    force_terminal=tty,
                    color_system="standard",
                    no_color=False,
                    legacy_windows=False,
                )
                original = CustomPrintHandler(console)
                with console.capture() as captured:
                    original.handle(record())
                native = captured.get()
                wrapped, wrapper_errors = adapter(original)
                with console.capture() as captured:
                    console.out("before", highlight=False)
                    wrapped.handle(record())
                    console.out("after", highlight=False)
                self.assertEqual(
                    captured.get(), "before\nattributed " + native.rstrip("\n") + "\nafter\n"
                )
                self.assertEqual(sink.getvalue(), "")
                self.assertEqual(original.errors, [])
                self.assertEqual(wrapper_errors, [])
                self.assertIs(original.console, console)

    def test_native_render_hook_redacts_or_suppresses_exactly_once(self):
        from rich.console import RenderHook
        from rich.text import Text

        for suppress in (False, True):
            for tty in (False, True):
                with self.subTest(suppress=suppress, tty=tty):
                    calls = []

                    class PrivacyHook(RenderHook):
                        def process_renderables(self, renderables, calls=calls, suppress=suppress):
                            calls.append(renderables)
                            return [] if suppress else [Text("redacted")]

                    sink = io.StringIO()
                    console = Console(file=sink, force_terminal=tty, color_system=None)
                    original = NativeHandler(console)
                    wrapped, wrapper_errors = adapter(original)
                    console.push_render_hook(PrivacyHook())
                    wrapped.handle(record())
                    console.pop_render_hook()
                    self.assertEqual(len(calls), 1)
                    self.assertEqual(sink.getvalue(), "" if suppress else "attributed redacted\n")
                    self.assertEqual(original.errors, [])
                    self.assertEqual(wrapper_errors, [])
                    self.assertIs(original.console, console)

    def test_forced_color_record_export_retains_styled_segment_metadata(self):
        from rich.text import Text

        class StyledHandler(NativeHandler):
            def emit(self, effective):
                self.console.print(
                    Text("red\nsecond", style="bold red"),
                    Text("blue", style="italic blue"),
                    sep="|",
                    soft_wrap=True,
                )

        sink = io.StringIO()
        console = Console(
            file=sink,
            record=True,
            width=10,
            force_terminal=True,
            color_system="standard",
            no_color=False,
            legacy_windows=False,
        )
        original = StyledHandler(console)
        with console.capture() as captured:
            original.handle(record())
        native = captured.get()
        self.assertIn("\x1b[", native)
        expected = "attributed " + native.rstrip("\n") + "\n"
        wrapped, wrapper_errors = adapter(original)
        wrapped.handle(record())
        self.assertFalse(
            any(segment.control and not segment.text for segment in console._record_buffer)
        )
        self.assertEqual(sink.getvalue(), expected)
        exported = console.export_text(styles=False)
        self.assertEqual(exported, "attributed red\nsecond|blue\n")
        self.assertNotIn("\x1b", exported)
        self.assertEqual(original.errors, [])
        self.assertEqual(wrapper_errors, [])
        self.assertIs(original.console, console)

    def test_hook_failure_keeps_enclosing_buffer_and_native_error_authority(self):
        from rich.console import RenderHook

        failure = DeliveryFailure("synthetic hook failure")

        class FailingHook(RenderHook):
            def process_renderables(self, renderables):
                raise failure

        sink = io.StringIO()
        console = Console(file=sink, force_terminal=False)
        original = NativeHandler(console)
        wrapped, wrapper_errors = adapter(original)
        with console.capture() as captured:
            console.out("before", highlight=False)
            console.push_render_hook(FailingHook())
            wrapped.handle(record())
            console.pop_render_hook()
            console.out("after", highlight=False)
        self.assertEqual(captured.get(), "before\nafter\n")
        self.assertEqual(sink.getvalue(), "")
        self.assertEqual(len(original.errors), 1)
        self.assertEqual(wrapper_errors, [])
        self.assertIs(original.console, console)
        self.assert_released(original, wrapped)

    def test_shared_console_keeps_other_threads_capture_independent(self):
        from rich.text import Text

        rendering = threading.Event()
        resume = threading.Event()

        class PausedRenderable:
            def __rich_console__(self, console, options):
                rendering.set()
                if not resume.wait(timeout=5):
                    raise DeliveryFailure("synthetic render wait expired")
                yield Text("thread diagnostic")

        class PausedHandler(NativeHandler):
            def emit(self, effective):
                self.console.print(PausedRenderable())

        sink = io.StringIO()
        console = Console(file=sink, force_terminal=False, color_system=None)
        original = PausedHandler(console)
        wrapped, wrapper_errors = adapter(original)
        worker_output = []

        def emit_from_worker():
            with console.capture() as captured:
                console.out("worker prefix", highlight=False)
                wrapped.handle(record())
                console.out("worker suffix", highlight=False)
            worker_output.append(captured.get())

        worker = threading.Thread(target=emit_from_worker)
        worker.start()
        try:
            self.assertTrue(rendering.wait(timeout=5))
            with console.capture() as captured:
                console.out("other prefix", highlight=False)
                console.out("other suffix", highlight=False)
            self.assertEqual(captured.get(), "other prefix\nother suffix\n")
        finally:
            resume.set()
            worker.join(timeout=5)
        self.assertFalse(worker.is_alive())
        self.assertEqual(
            worker_output,
            ["worker prefix\nattributed thread diagnostic\nworker suffix\n"],
        )
        self.assertEqual(sink.getvalue(), "")
        self.assertEqual(original.errors, [])
        self.assertEqual(wrapper_errors, [])
        self.assertIs(original.console, console)

    def test_live_host_controls_and_status_stay_outside_attribution(self):
        from rich.live import Live

        sink = io.StringIO()
        console = Console(
            file=sink,
            width=80,
            force_terminal=True,
            force_interactive=True,
            color_system="standard",
            no_color=False,
            legacy_windows=False,
            _environ={"TERM": "xterm"},
        )
        original = NativeHandler(console)
        wrapped, wrapper_errors = actual_adapter(original, tty=True)
        live = Live(
            "live status",
            console=console,
            auto_refresh=False,
            redirect_stdout=False,
            redirect_stderr=False,
        )
        live.start(refresh=True)
        try:
            sink.seek(0)
            sink.truncate()
            wrapped.handle(record())
            emitted = sink.getvalue()
        finally:
            live.stop()
        self.assertTrue(emitted.startswith("\r\x1b[2K"), repr(emitted))
        self.assertTrue(emitted.endswith("live status"), repr(emitted))
        self.assertEqual(emitted.count("live status"), 1)
        self.assertIn("retained diagnostic", emitted)
        self.assertEqual(original.errors, [])
        self.assertEqual(wrapper_errors, [])
        self.assertIs(original.console, console)

    def test_declared_tty_formatter_preserves_native_record_style_metadata(self):
        from rich.style import Style
        from rich.text import Text

        class StyledHandler(NativeHandler):
            def emit(self, effective):
                self.console.print(
                    Text("retained diagnostic", style=Style(color="#123456", bold=True))
                )

        for mode in ("no_color", "null_color"):
            with self.subTest(mode=mode):
                sink = io.StringIO()
                console = Console(
                    file=sink,
                    force_terminal=True,
                    record=True,
                    color_system=None if mode == "null_color" else "standard",
                    no_color=mode == "no_color",
                    legacy_windows=False,
                )
                original = StyledHandler(console)
                wrapped, wrapper_errors = actual_adapter(original, tty=True)
                if not getattr(wrapped.identity_formatter, "tty", False):
                    self.skipTest("This adapter's formatter uses JSON output")
                original.handle(record())
                native_emitted = sink.getvalue()
                native_recorded = console.export_text(styles=True)
                sink.seek(0)
                sink.truncate()
                wrapped.handle(record())
                self.assertTrue(sink.getvalue().endswith(native_emitted))
                self.assertTrue(console.export_text(styles=True).endswith(native_recorded))
                self.assertEqual(original.errors, [])
                self.assertEqual(wrapper_errors, [])
                self.assertIs(original.console, console)

    def test_console_subclass_super_print_updates_original_state_once(self):
        class StatefulConsole(Console):
            print_calls = 0

            def print(self, *objects, **kwargs):
                self.print_calls += 1
                return super().print(*objects, **kwargs)

        sink = io.StringIO()
        console = StatefulConsole(file=sink, force_terminal=False)
        original = NativeHandler(console)
        wrapped, wrapper_errors = adapter(original)
        wrapped.handle(record())
        self.assertEqual(console.print_calls, 1)
        self.assertIn("attributed retained diagnostic", sink.getvalue())
        self.assertEqual(original.errors, [])
        self.assertEqual(wrapper_errors, [])
        self.assertIs(original.console, console)

    def test_actual_formatter_retains_typed_json_payload(self):
        import json

        sink = io.StringIO()
        console = Console(file=sink, force_terminal=False, color_system=None)
        original = NativeHandler(console)
        wrapped, wrapper_errors = actual_adapter(original, tty=True)
        effective = record()
        effective.msg = '{"typed": 7, "nested": {"truth": true}}'
        wrapped.handle(effective)
        payload = json.loads(sink.getvalue())
        self.assertEqual(payload["typed"], 7)
        self.assertEqual(payload["nested"], {"truth": True})
        self.assertEqual(original.errors, [])
        self.assertEqual(wrapper_errors, [])

    def test_console_subclass_can_install_a_privacy_hook_before_super_print(self):
        from rich.console import RenderHook
        from rich.text import Text

        calls = []

        class PrivacyHook(RenderHook):
            def process_renderables(self, renderables):
                calls.append(renderables)
                return [Text("redacted")]

        class GuardedConsole(Console):
            def print(self, *objects, **kwargs):
                self.push_render_hook(PrivacyHook())
                try:
                    return super().print(*objects, **kwargs)
                finally:
                    self.pop_render_hook()

        sink = io.StringIO()
        console = GuardedConsole(file=sink, force_terminal=False)
        original_hooks = console._render_hooks[:]
        original = NativeHandler(console)
        wrapped, wrapper_errors = adapter(original)
        wrapped.handle(record())
        self.assertEqual(sink.getvalue(), "attributed redacted\n")
        self.assertEqual(len(calls), 1)
        self.assertEqual(console._render_hooks, original_hooks)
        self.assertEqual(original.errors, [])
        self.assertEqual(wrapper_errors, [])
        self.assertIs(original.console, console)


if __name__ == "__main__":
    unittest.main()
