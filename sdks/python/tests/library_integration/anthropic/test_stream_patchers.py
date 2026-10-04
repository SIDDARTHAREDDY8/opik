from dataclasses import dataclass
from typing import Callable, Iterator, List, Tuple
from unittest import mock
import asyncio
import json
import time

import pytest

import anthropic

import opik.integrations.anthropic.stream_patchers as sp

try:  # anthropic>=1.11 vendors httpx as httpx2; older releases use plain httpx
    import httpx2 as _httpx
except ImportError:  # pragma: no cover
    import httpx as _httpx


@dataclass(frozen=True)
class WrapperConfig:
    """Describes one of the six stream wrappers in stream_patchers.py."""

    id: str
    patch_fn: Callable
    global_name: str
    stream_cls: type
    patch_arg_cls: type
    needs_get_final_message: bool


def _sync_wrappers() -> list[WrapperConfig]:
    wrappers = [
        WrapperConfig(
            id="Stream",
            patch_fn=sp.patch_sync_stream,
            global_name="original_stream_iter_method",
            stream_cls=anthropic.Stream,
            patch_arg_cls=anthropic.Stream,
            needs_get_final_message=False,
        ),
        WrapperConfig(
            id="MessageStream",
            patch_fn=sp.patch_sync_message_stream_manager,
            global_name="original_message_stream_iter_method",
            stream_cls=anthropic.MessageStream,
            patch_arg_cls=anthropic.MessageStreamManager,
            needs_get_final_message=True,
        ),
    ]
    if sp.BetaMessageStream is not None:
        wrappers.append(
            WrapperConfig(
                id="BetaMessageStream",
                patch_fn=sp.patch_sync_beta_message_stream_manager,
                global_name="original_beta_message_stream_iter_method",
                stream_cls=sp.BetaMessageStream,
                patch_arg_cls=sp.BetaMessageStreamManager,
                needs_get_final_message=True,
            )
        )
    return wrappers


def _async_wrappers() -> list[WrapperConfig]:
    wrappers = [
        WrapperConfig(
            id="AsyncStream",
            patch_fn=sp.patch_async_stream,
            global_name="original_async_stream_aiter_method",
            stream_cls=anthropic.AsyncStream,
            patch_arg_cls=anthropic.AsyncStream,
            needs_get_final_message=False,
        ),
        WrapperConfig(
            id="AsyncMessageStream",
            patch_fn=sp.patch_async_message_stream_manager,
            global_name="original_async_message_stream_aiter_method",
            stream_cls=anthropic.AsyncMessageStream,
            patch_arg_cls=anthropic.AsyncMessageStreamManager,
            needs_get_final_message=True,
        ),
    ]
    if sp.BetaAsyncMessageStream is not None:
        wrappers.append(
            WrapperConfig(
                id="BetaAsyncMessageStream",
                patch_fn=sp.patch_async_beta_message_stream_manager,
                global_name="original_beta_async_message_stream_aiter_method",
                stream_cls=sp.BetaAsyncMessageStream,
                patch_arg_cls=sp.BetaAsyncMessageStreamManager,
                needs_get_final_message=True,
            )
        )
    return wrappers


def _raising_iter(self):
    raise RuntimeError("stream-blew-up")
    yield  # make this a generator function


async def _raising_aiter(self):
    raise RuntimeError("stream-blew-up")
    yield  # make this an async generator function


def _assert_error_info_matches(error_info):
    """Assert the callback's error_info reflects the injected RuntimeError.

    Guards against a wrapper reporting incorrect exception metadata (wrong
    type, dropped message, missing traceback) while still passing.
    """
    assert error_info is not None
    assert error_info["exception_type"] == "RuntimeError"
    assert error_info["message"] == "stream-blew-up"
    assert "test_stream_patchers.py" in error_info["traceback"]


@pytest.fixture
def restore_stream_patches():
    """Save and restore all class-level dunder methods and module globals
    that the stream patchers modify, so patches never leak across tests."""
    classes = [
        anthropic.Stream,
        anthropic.AsyncStream,
        anthropic.MessageStream,
        anthropic.AsyncMessageStream,
        anthropic.MessageStreamManager,
        anthropic.AsyncMessageStreamManager,
    ]
    if sp.BetaMessageStream is not None:
        classes += [
            sp.BetaMessageStream,
            sp.BetaAsyncMessageStream,
            sp.BetaMessageStreamManager,
            sp.BetaAsyncMessageStreamManager,
        ]

    saved_methods = {}
    for cls in classes:
        for name in ("__iter__", "__aiter__", "__enter__", "__aenter__"):
            if hasattr(cls, name):
                saved_methods[(cls, name)] = getattr(cls, name)

    saved_globals = {k: getattr(sp, k) for k in dir(sp) if k.startswith("original_")}

    yield

    for (cls, name), method in saved_methods.items():
        setattr(cls, name, method)
    for key, value in saved_globals.items():
        setattr(sp, key, value)


def _install(config: WrapperConfig, raising_fn: Callable, callback: mock.Mock):
    """Install a stream patcher's class-level override backed by raising_fn."""
    setattr(sp, config.global_name, raising_fn)
    throwaway = object.__new__(config.patch_arg_cls)
    config.patch_fn(
        throwaway,
        span_to_end=None,
        trace_to_end=None,
        finally_callback=callback,
    )


def _make_stream(config: WrapperConfig, tracked: bool, is_async: bool = False):
    stream = object.__new__(config.stream_cls)
    if tracked:
        stream.opik_tracked_instance = True
        stream.span_to_end = None
        stream.trace_to_end = None
        if config.needs_get_final_message:
            if is_async:

                async def _gfm():
                    return None

                stream.get_final_message = _gfm
            else:
                stream.get_final_message = lambda: None
    return stream


@pytest.mark.parametrize("config", _sync_wrappers(), ids=lambda c: c.id)
def test_sync_non_tracked_exception_propagates(restore_stream_patches, config):
    """Regression test for the `return` inside `finally` bug.

    Once a stream patcher installs its class-level __iter__ override, a
    non-tracked stream whose iteration raises must propagate the exception
    (the old `return` in `finally` silently swallowed it). The cleanup
    callback must not run for a stream opik never tracked.
    """
    callback = mock.Mock()
    _install(config, _raising_iter, callback)
    stream = _make_stream(config, tracked=False)

    with pytest.raises(RuntimeError, match="stream-blew-up"):
        for _ in stream:
            pass

    callback.assert_not_called()


@pytest.mark.parametrize("config", _sync_wrappers(), ids=lambda c: c.id)
def test_sync_tracked_exception_propagates_and_callback_runs(
    restore_stream_patches, config
):
    """A tracked stream that errors must propagate the exception AND run the
    span-closing callback exactly once with error_info set.
    """
    callback = mock.Mock()
    _install(config, _raising_iter, callback)
    stream = _make_stream(config, tracked=True)

    with pytest.raises(RuntimeError, match="stream-blew-up"):
        for _ in stream:
            pass

    callback.assert_called_once()
    _, kwargs = callback.call_args
    assert kwargs["capture_output"] is True
    _assert_error_info_matches(kwargs["error_info"])


@pytest.mark.parametrize("config", _async_wrappers(), ids=lambda c: c.id)
@pytest.mark.asyncio
async def test_async_non_tracked_exception_propagates(restore_stream_patches, config):
    """Async variant of the regression test — non-tracked async stream
    exceptions must propagate, cleanup callback must not run.
    """
    callback = mock.Mock()
    _install(config, _raising_aiter, callback)
    stream = _make_stream(config, tracked=False, is_async=True)

    with pytest.raises(RuntimeError, match="stream-blew-up"):
        async for _ in stream:
            pass

    callback.assert_not_called()


@pytest.mark.parametrize("config", _async_wrappers(), ids=lambda c: c.id)
@pytest.mark.asyncio
async def test_async_tracked_exception_propagates_and_callback_runs(
    restore_stream_patches, config
):
    """Async variant — tracked stream exceptions must propagate AND run the
    span-closing callback exactly once with error_info set.
    """
    callback = mock.Mock()
    _install(config, _raising_aiter, callback)
    stream = _make_stream(config, tracked=True, is_async=True)

    with pytest.raises(RuntimeError, match="stream-blew-up"):
        async for _ in stream:
            pass

    callback.assert_called_once()
    _, kwargs = callback.call_args
    assert kwargs["capture_output"] is True
    _assert_error_info_matches(kwargs["error_info"])


def _raw_events() -> list:
    """A minimal but complete raw event sequence, including tool input deltas
    that anthropic accumulates into partial JSON."""
    message = anthropic.types.Message.construct(
        id="msg_1",
        type="message",
        role="assistant",
        model="claude-3-5-sonnet-20241022",
        content=[],
        stop_reason=None,
        stop_sequence=None,
        usage={"input_tokens": 10, "output_tokens": 0},
    )
    types = anthropic.types

    return [
        types.RawMessageStartEvent.construct(type="message_start", message=message),
        types.RawContentBlockStartEvent.construct(
            type="content_block_start",
            index=0,
            content_block=types.TextBlock.construct(type="text", text=""),
        ),
        types.RawContentBlockDeltaEvent.construct(
            type="content_block_delta",
            index=0,
            delta=types.TextDelta.construct(type="text_delta", text="Hello"),
        ),
        types.RawContentBlockStopEvent.construct(type="content_block_stop", index=0),
        types.RawContentBlockStartEvent.construct(
            type="content_block_start",
            index=1,
            content_block=types.ToolUseBlock.construct(
                type="tool_use", id="tool_1", name="get_weather", input={}
            ),
        ),
        types.RawContentBlockDeltaEvent.construct(
            type="content_block_delta",
            index=1,
            delta=types.InputJSONDelta.construct(
                type="input_json_delta", partial_json='{"city": '
            ),
        ),
        types.RawContentBlockDeltaEvent.construct(
            type="content_block_delta",
            index=1,
            delta=types.InputJSONDelta.construct(
                type="input_json_delta", partial_json='"Paris"}'
            ),
        ),
        types.RawContentBlockStopEvent.construct(type="content_block_stop", index=1),
        types.RawMessageStopEvent.construct(type="message_stop"),
    ]


def _assert_accumulated_message_matches(output):
    assert output is not None
    assert output.content[0].text == "Hello"
    assert output.content[1].input == {"city": "Paris"}


def test_sync_stream_accumulates_events_into_the_final_message(
    restore_stream_patches,
):
    """Regression test for anthropic changing `accumulate_event()`'s signature
    (1.5.0 made the partial-JSON buffer a required caller-owned argument).
    """
    events = _raw_events()

    def _iter_events(self):
        yield from events

    callback = mock.Mock()
    config = next(c for c in _sync_wrappers() if c.id == "Stream")
    _install(config, _iter_events, callback)
    stream = _make_stream(config, tracked=True)

    assert list(stream) == events

    callback.assert_called_once()
    _, kwargs = callback.call_args
    assert kwargs["error_info"] is None
    _assert_accumulated_message_matches(kwargs["output"])


@pytest.mark.asyncio
async def test_async_stream_accumulates_events_into_the_final_message(
    restore_stream_patches,
):
    """Async variant of the `accumulate_event()` signature regression test."""
    events = _raw_events()

    async def _aiter_events(self):
        for event in events:
            yield event

    callback = mock.Mock()
    config = next(c for c in _async_wrappers() if c.id == "AsyncStream")
    _install(config, _aiter_events, callback)
    stream = _make_stream(config, tracked=True, is_async=True)

    assert [event async for event in stream] == events

    callback.assert_called_once()
    _, kwargs = callback.call_args
    assert kwargs["error_info"] is None
    _assert_accumulated_message_matches(kwargs["output"])


# ---------------------------------------------------------------------------
# Early-exit regression tests for https://github.com/comet-ml/opik/issues/8719
#
# Breaking out of `for event in stream` on a tracked `messages.stream()` must
# not drain the rest of the response: the span-closing callback must run
# exactly once and log only the message accumulated from the events the
# caller actually read. The async variant must also end the span when the
# response is already closed (previously `await get_final_message()` raised
# httpx.ReadError in the generator finalizer and the span never ended).
# ---------------------------------------------------------------------------


def _message_stream_configs() -> List[WrapperConfig]:
    return [
        c for c in _sync_wrappers() + _async_wrappers() if c.needs_get_final_message
    ]


def _snapshot_attr(config: WrapperConfig) -> str:
    # Mirrors the name-mangled `__final_message_snapshot` attribute that
    # anthropic's MessageStream.__init__ always initializes to None.
    return f"_{config.stream_cls.__name__}__final_message_snapshot"


def _make_tracked_message_stream(
    config: WrapperConfig,
    callback: mock.Mock,
    iter_fn: Callable,
    is_async: bool,
    snapshot: object = None,
) -> object:
    """Install the patcher and build a tracked stream whose `get_final_message`
    explodes if called, proving the wrapper never drains the stream."""
    _install(config, iter_fn, callback)
    stream = _make_stream(config, tracked=True, is_async=is_async)
    setattr(stream, _snapshot_attr(config), snapshot)
    if is_async:

        async def _forbidden_get_final_message():
            raise AssertionError("get_final_message() must not drain the stream")

        stream.get_final_message = _forbidden_get_final_message
    else:
        stream.get_final_message = mock.Mock(
            side_effect=AssertionError("get_final_message() must not drain the stream")
        )
    return stream


def _counting_iter(events: list, pulled: list) -> Callable:
    def _iter(self):
        for event in events:
            pulled.append(event)
            yield event

    return _iter


def _counting_aiter(events: list, pulled: list) -> Callable:
    async def _aiter(self):
        for event in events:
            pulled.append(event)
            yield event

    return _aiter


@pytest.mark.parametrize(
    "config",
    [c for c in _sync_wrappers() if c.needs_get_final_message],
    ids=lambda c: c.id,
)
def test_sync_message_stream_break_does_not_drain_and_logs_what_was_read(
    restore_stream_patches, config
):
    events = [object() for _ in range(40)]
    pulled: list = []
    snapshot = object()
    callback = mock.Mock()
    stream = _make_tracked_message_stream(
        config,
        callback,
        _counting_iter(events, pulled),
        is_async=False,
        snapshot=snapshot,
    )

    for _event in stream:
        break

    assert pulled == events[:1]
    stream.get_final_message.assert_not_called()
    callback.assert_called_once()
    _, kwargs = callback.call_args
    assert kwargs["error_info"] is None
    assert kwargs["output"] is snapshot


@pytest.mark.parametrize(
    "config",
    [c for c in _async_wrappers() if c.needs_get_final_message],
    ids=lambda c: c.id,
)
@pytest.mark.asyncio
async def test_async_message_stream_break_does_not_drain_and_logs_what_was_read(
    restore_stream_patches, config
):
    events = [object() for _ in range(40)]
    pulled: list = []
    snapshot = object()
    callback = mock.Mock()
    stream = _make_tracked_message_stream(
        config,
        callback,
        _counting_aiter(events, pulled),
        is_async=True,
        snapshot=snapshot,
    )

    drain_error = None
    try:
        aiter = stream.__aiter__()
        async for _event in aiter:
            break
        # NOTE: `async for ... break` finalizes the async generator through the
        # event loop's asyncgen hooks, which is timing-dependent; aclose() runs
        # the very same finally block deterministically.
        await aiter.aclose()
    except Exception as exc:  # noqa: BLE001 -- only the buggy code leaks here
        drain_error = exc

    assert drain_error is None, (
        f"early exit must not drain the stream, got {drain_error!r}"
    )
    assert pulled == events[:1]
    callback.assert_called_once()
    _, kwargs = callback.call_args
    assert kwargs["error_info"] is None
    assert kwargs["output"] is snapshot


@pytest.mark.parametrize(
    "config",
    [c for c in _sync_wrappers() if c.needs_get_final_message],
    ids=lambda c: c.id,
)
def test_sync_message_stream_error_in_loop_reports_error_without_drain(
    restore_stream_patches, config
):
    events = [object() for _ in range(40)]
    pulled: list = []

    def _iter(self):
        for i, event in enumerate(events):
            pulled.append(event)
            yield event
            if i == 1:
                raise RuntimeError("boom")

    callback = mock.Mock()
    stream = _make_tracked_message_stream(
        config, callback, _iter, is_async=False, snapshot=object()
    )

    with pytest.raises(RuntimeError, match="boom"):
        for _event in stream:
            pass

    assert pulled == events[:2]
    stream.get_final_message.assert_not_called()
    callback.assert_called_once()
    _, kwargs = callback.call_args
    error_info = kwargs["error_info"]
    assert error_info is not None
    assert error_info["exception_type"] == "RuntimeError"
    assert error_info["message"] == "boom"
    assert kwargs["output"] is None


@pytest.mark.parametrize(
    "config",
    [c for c in _async_wrappers() if c.needs_get_final_message],
    ids=lambda c: c.id,
)
@pytest.mark.asyncio
async def test_async_message_stream_error_in_loop_reports_error_without_drain(
    restore_stream_patches, config
):
    events = [object() for _ in range(40)]
    pulled: list = []

    async def _aiter(self):
        for i, event in enumerate(events):
            pulled.append(event)
            yield event
            if i == 1:
                raise RuntimeError("boom")

    callback = mock.Mock()
    stream = _make_tracked_message_stream(
        config, callback, _aiter, is_async=True, snapshot=object()
    )

    with pytest.raises(RuntimeError, match="boom"):
        async for _event in stream:
            pass

    assert pulled == events[:2]
    callback.assert_called_once()
    _, kwargs = callback.call_args
    error_info = kwargs["error_info"]
    assert error_info is not None
    assert error_info["exception_type"] == "RuntimeError"
    assert error_info["message"] == "boom"
    assert kwargs["output"] is None


@pytest.mark.parametrize("config", _message_stream_configs(), ids=lambda c: c.id)
@pytest.mark.asyncio
async def test_message_stream_consumed_without_events_logs_nothing_instead_of_crashing(
    restore_stream_patches, config
):
    """No event read yet means `current_message_snapshot` asserts; the wrapper
    must log None (not crash, not drain) and still end the span."""
    is_async = config in _async_wrappers()

    def _iter(self):
        return
        yield  # make this a generator function

    async def _aiter(self):
        return
        yield  # make this an async generator function

    callback = mock.Mock()
    stream = _make_tracked_message_stream(
        config,
        callback,
        _aiter if is_async else _iter,
        is_async=is_async,
        snapshot=None,
    )

    if is_async:
        assert [event async for event in stream] == []
    else:
        assert list(stream) == []

    callback.assert_called_once()
    _, kwargs = callback.call_args
    assert kwargs["error_info"] is None
    assert kwargs["output"] is None


def _sse_event(name: str, data: dict) -> bytes:
    return f"event: {name}\ndata: {json.dumps(data)}\n\n".encode()


def _sse_event_stream(
    pulled: list, n_deltas: int = 40
) -> Iterator[Tuple[float, bytes]]:
    """Yield (delay, chunk) pairs; `pulled` records deltas as they are consumed."""
    yield (
        0.0,
        _sse_event(
            "message_start",
            {
                "type": "message_start",
                "message": {
                    "id": "m",
                    "type": "message",
                    "role": "assistant",
                    "model": "m",
                    "content": [],
                    "stop_reason": None,
                    "stop_sequence": None,
                    "usage": {"input_tokens": 3, "output_tokens": 1},
                },
            },
        ),
    )
    yield (
        0.0,
        _sse_event(
            "content_block_start",
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "text", "text": ""},
            },
        ),
    )
    for i in range(n_deltas):
        pulled.append(i)
        yield (
            0.025,
            _sse_event(
                "content_block_delta",
                {
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {"type": "text_delta", "text": f"w{i} "},
                },
            ),
        )
    yield (
        0.0,
        _sse_event("content_block_stop", {"type": "content_block_stop", "index": 0}),
    )
    yield 0.0, _sse_event("message_stop", {"type": "message_stop"})


def _sse_body(pulled: list, n_deltas: int = 40):
    def body():
        for delay, chunk in _sse_event_stream(pulled, n_deltas):
            if delay:
                time.sleep(delay)
            yield chunk

    return body


def _sse_abody(pulled: list, n_deltas: int = 40):
    async def abody():
        for delay, chunk in _sse_event_stream(pulled, n_deltas):
            if delay:
                await asyncio.sleep(delay)
            yield chunk

    return abody


def _mock_transport_client(pulled: list, async_client: bool = False):
    if async_client:
        content: object = _sse_abody(pulled)()
    else:
        content = _sse_body(pulled)()
    transport = _httpx.MockTransport(
        lambda r: _httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=content,  # type: ignore[arg-type]
        )
    )
    if async_client:
        return anthropic.AsyncAnthropic(
            api_key="x",
            base_url="http://mock",
            http_client=_httpx.AsyncClient(transport=transport),
        )
    return anthropic.Anthropic(
        api_key="x",
        base_url="http://mock",
        http_client=_httpx.Client(transport=transport),
    )


def test_sync_tracked_message_stream_break_matches_untracked_server_reads(
    restore_stream_patches,
):
    """End-to-end variant of the issue repro: a real MessageStream over a mock
    transport. Breaking out early must leave the server with ~1 delta read
    (the buggy code drains all 40) and the span-closing callback must run
    with the partial message the caller actually read."""
    pulled: list = []
    client = _mock_transport_client(pulled)
    callback = mock.Mock()
    manager = client.messages.stream(
        model="m", max_tokens=100, messages=[{"role": "user", "content": "hi"}]
    )
    sp.patch_sync_message_stream_manager(
        manager, span_to_end=None, trace_to_end=None, finally_callback=callback
    )

    with manager as stream:
        for event in stream:
            if event.type == "content_block_delta":
                break

    assert len(pulled) <= 2, f"stream was drained: server sent {len(pulled)}/40 deltas"
    callback.assert_called_once()
    _, kwargs = callback.call_args
    assert kwargs["error_info"] is None
    output = kwargs["output"]
    assert output is not None
    text = "".join(block.text for block in output.content if block.type == "text")
    assert text == "w0 ", f"span must log only what was read, got {text!r}"


@pytest.mark.asyncio
async def test_async_tracked_message_stream_break_matches_untracked_server_reads(
    restore_stream_patches,
):
    """Async end-to-end variant of the issue repro."""
    pulled: list = []
    client = _mock_transport_client(pulled, async_client=True)
    callback = mock.Mock()
    manager = client.messages.stream(
        model="m", max_tokens=100, messages=[{"role": "user", "content": "hi"}]
    )
    sp.patch_async_message_stream_manager(
        manager, span_to_end=None, trace_to_end=None, finally_callback=callback
    )

    async with manager as stream:
        aiter = stream.__aiter__()
        async for event in aiter:
            if event.type == "content_block_delta":
                break
        # See the note in test_async_message_stream_break_does_not_drain...:
        # aclose() runs the same finally block deterministically.
        await aiter.aclose()

    assert len(pulled) <= 2, f"stream was drained: server sent {len(pulled)}/40 deltas"
    callback.assert_called_once()
    _, kwargs = callback.call_args
    assert kwargs["error_info"] is None
    output = kwargs["output"]
    assert output is not None
    text = "".join(block.text for block in output.content if block.type == "text")
    assert text == "w0 ", f"span must log only what was read, got {text!r}"
