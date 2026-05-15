import asyncio
from collections.abc import Callable, Iterable
from dataclasses import dataclass
import inspect
import json
from uuid import uuid4

import numpy as np
import sounddevice as sd
import websockets

from config import TTSSettings


EVENT_START_CONNECTION = 1
EVENT_FINISH_CONNECTION = 2
EVENT_CONNECTION_STARTED = 50
EVENT_CONNECTION_FAILED = 51
EVENT_CONNECTION_FINISHED = 52
EVENT_START_SESSION = 100
EVENT_CANCEL_SESSION = 101
EVENT_FINISH_SESSION = 102
EVENT_SESSION_STARTED = 150
EVENT_SESSION_CANCELED = 151
EVENT_SESSION_FINISHED = 152
EVENT_SESSION_FAILED = 153
EVENT_TASK_REQUEST = 200
EVENT_TTS_RESPONSE = 352

MESSAGE_TYPE_FULL_CLIENT_REQUEST = 0x1
MESSAGE_TYPE_FULL_SERVER_RESPONSE = 0x9
MESSAGE_TYPE_AUDIO_ONLY_RESPONSE = 0xB
MESSAGE_TYPE_ERROR = 0xF
FLAG_WITH_EVENT = 0x4
SERIALIZATION_RAW = 0x0
SERIALIZATION_JSON = 0x1
COMPRESSION_NONE = 0x0
PROTOCOL_VERSION = 0x1
HEADER_SIZE_WORDS = 0x1
MAX_ERROR_DETAIL_LENGTH = 300


class TTSError(RuntimeError):
    """User-facing error for text-to-speech failures."""


class TextToSpeech:
    """Interface for streaming assistant text to audible speech."""

    def speak_stream(
        self,
        text_chunks: Iterable[str],
        *,
        on_text: Callable[[str], None] | None = None,
    ) -> None:
        """Speak TEXT_CHUNKS while optionally echoing each chunk."""


class SilentTextToSpeech(TextToSpeech):
    """A no-op TTS provider used when speech playback is disabled."""

    def speak_stream(
        self,
        text_chunks: Iterable[str],
        *,
        on_text: Callable[[str], None] | None = None,
    ) -> None:
        """Consume text chunks without producing audio."""
        for chunk in text_chunks:
            if on_text is not None:
                on_text(chunk)


class DoubaoStreamingTTS(TextToSpeech):
    """Stream LLM text into Doubao bidirectional TTS and play PCM audio."""

    def __init__(self, settings: TTSSettings) -> None:
        """Store validated Doubao TTS settings."""
        self._settings = settings

    def speak_stream(
        self,
        text_chunks: Iterable[str],
        *,
        on_text: Callable[[str], None] | None = None,
    ) -> None:
        """Run one streaming TTS session for TEXT_CHUNKS."""
        try:
            asyncio.run(self._speak_stream_async(text_chunks, on_text=on_text))
        except TTSError:
            raise
        except (OSError, sd.PortAudioError, websockets.WebSocketException) as exc:
            raise TTSError(f"TTS playback failed: {exc}") from exc

    async def _speak_stream_async(
        self,
        text_chunks: Iterable[str],
        *,
        on_text: Callable[[str], None] | None,
    ) -> None:
        """Connect, send text chunks, receive audio, and close cleanly."""
        headers = _build_headers(self._settings)
        try:
            async with websockets.connect(
                self._settings.endpoint,
                **_websocket_connect_kwargs(headers, self._settings.connect_timeout),
            ) as websocket:
                await self._start_connection(websocket)
                session_id = uuid4().hex
                await self._start_session(websocket, session_id)

                with PCMStreamPlayer(self._settings) as player:
                    receiver = asyncio.create_task(
                        self._receive_audio(websocket, session_id, player)
                    )
                    sender = asyncio.create_task(
                        self._send_text_chunks(
                            websocket,
                            session_id,
                            text_chunks,
                            on_text=on_text,
                        )
                    )
                    await self._wait_for_session(sender, receiver, websocket, session_id)

                await self._finish_connection(websocket)
        except OSError as exc:
            raise TTSError(f"Cannot connect to Doubao TTS: {exc}") from exc

    async def _start_connection(self, websocket) -> None:
        """Send StartConnection and wait for ConnectionStarted."""
        await websocket.send(_build_connection_frame(EVENT_START_CONNECTION))
        frame = _parse_server_frame(await websocket.recv())
        if frame.event == EVENT_CONNECTION_FAILED:
            raise TTSError(_format_frame_error("Doubao TTS connection failed", frame))
        if frame.event != EVENT_CONNECTION_STARTED:
            raise TTSError(f"Unexpected TTS connection event: {frame.event}")

    async def _finish_connection(self, websocket) -> None:
        """Send FinishConnection and ignore a missing close acknowledgement."""
        try:
            await websocket.send(_build_connection_frame(EVENT_FINISH_CONNECTION))
            frame = _parse_server_frame(await websocket.recv())
            if frame.event not in {EVENT_CONNECTION_FINISHED, None}:
                raise TTSError(f"Unexpected TTS close event: {frame.event}")
        except websockets.ConnectionClosed:
            return

    async def _start_session(self, websocket, session_id: str) -> None:
        """Send StartSession and wait for SessionStarted."""
        await websocket.send(
            _build_session_frame(
                EVENT_START_SESSION,
                session_id,
                _build_start_session_payload(self._settings),
            )
        )
        frame = _parse_server_frame(await websocket.recv())
        if frame.event == EVENT_SESSION_FAILED:
            raise TTSError(_format_frame_error("Doubao TTS session failed", frame))
        if frame.event != EVENT_SESSION_STARTED:
            raise TTSError(f"Unexpected TTS session event: {frame.event}")

    async def _send_text_chunks(
        self,
        websocket,
        session_id: str,
        text_chunks: Iterable[str],
        *,
        on_text: Callable[[str], None] | None,
    ) -> None:
        """Send each model chunk as a Doubao TaskRequest."""
        iterator = iter(text_chunks)
        while True:
            chunk = await asyncio.to_thread(_next_chunk, iterator)
            if chunk is None:
                break

            if on_text is not None:
                on_text(chunk)
            await websocket.send(_build_task_request_frame(session_id, chunk))

        await websocket.send(_build_session_frame(EVENT_FINISH_SESSION, session_id, {}))

    async def _receive_audio(
        self,
        websocket,
        session_id: str,
        player: "PCMStreamPlayer",
    ) -> None:
        """Receive Doubao frames until the current session is finished."""
        while True:
            frame = _parse_server_frame(await websocket.recv())
            if frame.session_id and frame.session_id != session_id:
                continue
            if frame.event in {EVENT_SESSION_FAILED, EVENT_SESSION_CANCELED}:
                raise TTSError(_format_frame_error("Doubao TTS session stopped", frame))
            if frame.event == EVENT_TTS_RESPONSE and frame.payload:
                await asyncio.to_thread(player.write, frame.payload)
                continue
            if frame.event == EVENT_SESSION_FINISHED:
                return

    async def _wait_for_session(
        self,
        sender: asyncio.Task,
        receiver: asyncio.Task,
        websocket,
        session_id: str,
    ) -> None:
        """Wait for sender and receiver, canceling the session on failure."""
        try:
            await asyncio.wait_for(
                asyncio.gather(sender, receiver),
                timeout=self._settings.session_timeout,
            )
        except asyncio.TimeoutError as exc:
            sender.cancel()
            receiver.cancel()
            await _cancel_session(websocket, session_id)
            raise TTSError("Doubao TTS session timed out.") from exc
        except Exception:
            sender.cancel()
            receiver.cancel()
            await _cancel_session(websocket, session_id)
            raise


class PCMStreamPlayer:
    """Play signed 16-bit PCM chunks with sounddevice."""

    def __init__(self, settings: TTSSettings) -> None:
        """Create a lazy output stream for SETTINGS."""
        self._settings = settings
        self._stream = None
        self._leftover = b""

    def __enter__(self) -> "PCMStreamPlayer":
        """Open the output stream."""
        self._stream = sd.OutputStream(
            samplerate=self._settings.sample_rate,
            channels=self._settings.channels,
            dtype="int16",
            device=self._settings.output_device,
        )
        self._stream.start()
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        """Flush and close the output stream."""
        if self._stream is None:
            return
        try:
            self._stream.stop()
        finally:
            self._stream.close()
            self._stream = None

    def write(self, audio: bytes) -> None:
        """Write complete PCM frames to the output stream."""
        if self._stream is None:
            raise TTSError("TTS audio output stream is not open.")

        frame_width = 2 * self._settings.channels
        data = self._leftover + audio
        complete_size = len(data) - (len(data) % frame_width)
        if complete_size <= 0:
            self._leftover = data
            return

        chunk = data[:complete_size]
        self._leftover = data[complete_size:]
        samples = np.frombuffer(chunk, dtype="<i2")
        samples = samples.reshape(-1, self._settings.channels)
        self._stream.write(samples)


@dataclass(frozen=True)
class ParsedFrame:
    """A parsed Doubao websocket frame."""

    event: int | None
    payload: bytes
    payload_json: object | None
    connection_id: str | None = None
    session_id: str | None = None
    error_code: int | None = None


def create_text_to_speech(settings: TTSSettings) -> TextToSpeech:
    """Create the configured TTS provider."""
    if settings.provider == "none":
        return SilentTextToSpeech()
    if settings.provider == "doubao":
        return DoubaoStreamingTTS(settings)
    raise TTSError(f"Unsupported TTS provider: {settings.provider}")


async def _cancel_session(websocket, session_id: str) -> None:
    """Best-effort cancellation for a failed TTS session."""
    try:
        await websocket.send(_build_session_frame(EVENT_CANCEL_SESSION, session_id, {}))
    except Exception:
        return


def _build_headers(settings: TTSSettings) -> dict[str, str]:
    """Return websocket headers for the new Volcengine console."""
    if not settings.api_key:
        raise TTSError("DOUBAO_TTS_API_KEY is required.")

    return {
        "X-Api-Key": settings.api_key,
        "X-Api-Resource-Id": settings.resource_id,
        "X-Api-Connect-Id": uuid4().hex,
    }


def _websocket_connect_kwargs(headers: dict[str, str], timeout: float) -> dict[str, object]:
    """Return websocket kwargs for both old and new websockets releases."""
    signature = inspect.signature(websockets.connect)
    header_name = (
        "additional_headers"
        if "additional_headers" in signature.parameters
        else "extra_headers"
    )
    return {
        header_name: headers,
        "open_timeout": timeout,
    }


def _build_start_session_payload(settings: TTSSettings) -> dict[str, object]:
    """Return Doubao session parameters that stay fixed for one session."""
    audio_params = {
        "format": settings.audio_format,
        "sample_rate": settings.sample_rate,
        "speech_rate": settings.speech_rate,
        "loudness_rate": settings.loudness_rate,
    }
    return {
        "event": EVENT_START_SESSION,
        "namespace": "BidirectionalTTS",
        "user": {"uid": settings.user_uid},
        "req_params": {
            "speaker": settings.speaker,
            "audio_params": audio_params,
        },
    }


def _build_task_request_frame(session_id: str, text: str) -> bytes:
    """Build a TaskRequest frame containing one text fragment."""
    payload = {
        "event": EVENT_TASK_REQUEST,
        "namespace": "BidirectionalTTS",
        "req_params": {"text": text},
    }
    return _build_session_frame(EVENT_TASK_REQUEST, session_id, payload)


def _build_connection_frame(event: int) -> bytes:
    """Build a connection-level JSON frame."""
    return _build_header() + _int32(event) + _payload_json({})


def _build_session_frame(event: int, session_id: str, payload: object) -> bytes:
    """Build a session-level JSON frame."""
    session_bytes = session_id.encode("utf-8")
    return (
        _build_header()
        + _int32(event)
        + _uint32(len(session_bytes))
        + session_bytes
        + _payload_json(payload)
    )


def _build_header() -> bytes:
    """Return the fixed v1 JSON full-client request header."""
    return bytes(
        [
            (PROTOCOL_VERSION << 4) | HEADER_SIZE_WORDS,
            (MESSAGE_TYPE_FULL_CLIENT_REQUEST << 4) | FLAG_WITH_EVENT,
            (SERIALIZATION_JSON << 4) | COMPRESSION_NONE,
            0,
        ]
    )


def _payload_json(payload: object) -> bytes:
    """Return a length-prefixed compact JSON payload."""
    payload_bytes = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode(
        "utf-8"
    )
    return _uint32(len(payload_bytes)) + payload_bytes


def _parse_server_frame(raw: bytes | str) -> ParsedFrame:
    """Parse one Doubao binary websocket frame."""
    if isinstance(raw, str):
        return ParsedFrame(event=None, payload=raw.encode("utf-8"), payload_json=raw)
    if len(raw) < 4:
        raise TTSError("Doubao TTS returned a malformed frame.")

    first, second, third, _ = raw[:4]
    header_size = (first & 0x0F) * 4
    message_type = second >> 4
    flags = second & 0x0F
    serialization = third >> 4
    offset = header_size

    if message_type == MESSAGE_TYPE_ERROR:
        return _parse_error_frame(raw, offset, serialization)

    event = None
    if flags & FLAG_WITH_EVENT:
        event, offset = _read_int32(raw, offset)

    connection_id = None
    session_id = None
    if event in {EVENT_CONNECTION_STARTED, EVENT_CONNECTION_FAILED, EVENT_CONNECTION_FINISHED}:
        connection_id, offset = _read_optional_id(raw, offset)
    elif event is not None:
        session_id, offset = _read_optional_id(raw, offset)

    payload, _ = _read_payload(raw, offset)
    payload_json = _decode_payload_json(payload, serialization)
    return ParsedFrame(
        event=event,
        payload=payload,
        payload_json=payload_json,
        connection_id=connection_id,
        session_id=session_id,
    )


def _parse_error_frame(raw: bytes, offset: int, serialization: int) -> ParsedFrame:
    """Parse a websocket error frame."""
    error_code, offset = _read_int32(raw, offset)
    payload = raw[offset:]
    return ParsedFrame(
        event=None,
        payload=payload,
        payload_json=_decode_payload_json(payload, serialization),
        error_code=error_code,
    )


def _read_optional_id(raw: bytes, offset: int) -> tuple[str | None, int]:
    """Read an optional length-prefixed id if one is present."""
    if len(raw) < offset + 4:
        return None, offset

    size, next_offset = _read_uint32(raw, offset)
    if size < 0 or len(raw) < next_offset + size:
        return None, offset

    value = raw[next_offset : next_offset + size].decode("utf-8", errors="replace")
    return value or None, next_offset + size


def _read_payload(raw: bytes, offset: int) -> tuple[bytes, int]:
    """Read a length-prefixed payload or return the remaining bytes."""
    if len(raw) < offset + 4:
        return raw[offset:], len(raw)

    size, next_offset = _read_uint32(raw, offset)
    if len(raw) < next_offset + size:
        return raw[offset:], len(raw)
    return raw[next_offset : next_offset + size], next_offset + size


def _decode_payload_json(payload: bytes, serialization: int) -> object | None:
    """Decode JSON payloads and leave audio payloads untouched."""
    if serialization != SERIALIZATION_JSON or not payload:
        return None
    try:
        return json.loads(payload.decode("utf-8"))
    except json.JSONDecodeError:
        return payload.decode("utf-8", errors="replace")


def _format_frame_error(prefix: str, frame: ParsedFrame) -> str:
    """Return a compact user-facing message for a failed TTS frame."""
    detail = frame.payload_json if frame.payload_json is not None else frame.payload
    if isinstance(detail, bytes):
        detail_text = detail.decode("utf-8", errors="replace")
    else:
        detail_text = json.dumps(detail, ensure_ascii=False)
    detail_text = " ".join(detail_text.split())
    if len(detail_text) > MAX_ERROR_DETAIL_LENGTH:
        detail_text = f"{detail_text[:MAX_ERROR_DETAIL_LENGTH]}..."
    if frame.error_code is not None:
        return f"{prefix}: error {frame.error_code}. {detail_text}"
    return f"{prefix}: {detail_text}"


def _next_chunk(iterator) -> str | None:
    """Return the next non-empty text chunk or None when the iterator ends."""
    for chunk in iterator:
        if chunk:
            return chunk
    return None


def _int32(value: int) -> bytes:
    """Encode VALUE as a signed big-endian int32."""
    return int(value).to_bytes(4, "big", signed=True)


def _uint32(value: int) -> bytes:
    """Encode VALUE as an unsigned big-endian uint32."""
    return int(value).to_bytes(4, "big", signed=False)


def _read_int32(raw: bytes, offset: int) -> tuple[int, int]:
    """Read a signed big-endian int32 from RAW."""
    if len(raw) < offset + 4:
        raise TTSError("Doubao TTS returned a truncated int32 field.")
    return int.from_bytes(raw[offset : offset + 4], "big", signed=True), offset + 4


def _read_uint32(raw: bytes, offset: int) -> tuple[int, int]:
    """Read an unsigned big-endian uint32 from RAW."""
    if len(raw) < offset + 4:
        raise TTSError("Doubao TTS returned a truncated uint32 field.")
    return int.from_bytes(raw[offset : offset + 4], "big", signed=False), offset + 4
