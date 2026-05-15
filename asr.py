from dataclasses import dataclass
import json
from pathlib import Path
from typing import Protocol
from urllib.error import HTTPError, URLError
from urllib.request import ProxyHandler, Request, build_opener
from uuid import uuid4

from config import ASRSettings


class ASRError(RuntimeError):
    """User-facing error for speech recognition failures."""


@dataclass(frozen=True)
class TranscriptionResult:
    """Text and lightweight metadata returned by a transcription provider."""

    text: str
    language: str | None
    duration: float | None


class SpeechRecognizer(Protocol):
    """Common interface implemented by ASR clients."""

    def transcribe(self, audio_path: Path) -> TranscriptionResult:
        """Return recognized text from AUDIO_PATH."""


class RemoteASR:
    """ASR client that uploads WAV files to a LAN transcription service."""

    def __init__(self, settings: ASRSettings) -> None:
        """Store remote endpoint settings from SETTINGS."""
        self._settings = settings
        if not settings.remote_url:
            raise ASRError("ASR_REMOTE_URL is required.")

    def transcribe(self, audio_path: Path) -> TranscriptionResult:
        """Upload AUDIO_PATH and return the parsed transcription response."""
        if not audio_path.exists():
            raise ASRError(f"Audio file does not exist: {audio_path}")

        request = self._build_request(audio_path)
        try:
            with _no_proxy_opener().open(
                request, timeout=self._settings.remote_timeout
            ) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            detail = _read_http_error_detail(exc)
            raise ASRError(f"Remote ASR returned HTTP {exc.code}. {detail}") from exc
        except URLError as exc:
            raise ASRError(f"Remote ASR connection failed: {exc.reason}") from exc
        except TimeoutError as exc:
            raise ASRError("Remote ASR request timed out.") from exc
        except json.JSONDecodeError as exc:
            raise ASRError("Remote ASR returned invalid JSON.") from exc

        return _parse_transcription_payload(payload)

    def _build_request(self, audio_path: Path) -> Request:
        """Build a multipart/form-data upload request for AUDIO_PATH."""
        boundary = f"voice-agent-{uuid4().hex}"
        body = _build_multipart_body(
            boundary=boundary,
            audio_path=audio_path,
            language=self._settings.language,
        )
        headers = {
            "Content-Type": f"multipart/form-data; boundary={boundary}",
            "Content-Length": str(len(body)),
        }
        if self._settings.remote_api_key:
            headers["Authorization"] = f"Bearer {self._settings.remote_api_key}"

        return Request(
            self._settings.remote_url,
            data=body,
            headers=headers,
            method="POST",
        )


def create_speech_recognizer(settings: ASRSettings) -> SpeechRecognizer:
    """Create the configured ASR client."""
    if settings.provider == "remote":
        return RemoteASR(settings)
    raise ASRError(f"Unsupported ASR provider: {settings.provider}")


def _build_multipart_body(
    *, boundary: str, audio_path: Path, language: str | None
) -> bytes:
    """Return a multipart body containing language and one WAV file."""
    parts: list[bytes] = []
    if language:
        parts.extend(
            [
                f"--{boundary}\r\n".encode("utf-8"),
                b'Content-Disposition: form-data; name="language"\r\n\r\n',
                language.encode("utf-8"),
                b"\r\n",
            ]
        )

    parts.extend(
        [
            f"--{boundary}\r\n".encode("utf-8"),
            (
                'Content-Disposition: form-data; name="file"; '
                f'filename="{audio_path.name}"\r\n'
            ).encode("utf-8"),
            b"Content-Type: audio/wav\r\n\r\n",
            audio_path.read_bytes(),
            b"\r\n",
            f"--{boundary}--\r\n".encode("utf-8"),
        ]
    )
    return b"".join(parts)


def _parse_transcription_payload(payload: object) -> TranscriptionResult:
    """Convert a remote JSON payload into TranscriptionResult."""
    if not isinstance(payload, dict):
        raise ASRError("Remote ASR returned a non-object JSON payload.")

    text = str(payload.get("text", "")).strip()
    if not text:
        raise ASRError("Remote ASR returned empty text.")

    return TranscriptionResult(
        text=text,
        language=_optional_str(payload.get("language")),
        duration=_optional_float(payload.get("duration")),
    )


def _no_proxy_opener():
    """Return an opener that ignores system proxy variables for LAN ASR."""
    return build_opener(ProxyHandler({}))


def _read_http_error_detail(exc: HTTPError) -> str:
    """Return a compact error detail from an HTTPError body."""
    try:
        raw = exc.read().decode("utf-8", errors="replace")
    except Exception:
        return "The server did not return readable details."

    detail = " ".join(raw.split())
    return detail or "The server did not return error details."


def _optional_str(value: object) -> str | None:
    """Return VALUE as a non-empty string when possible."""
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _optional_float(value: object) -> float | None:
    """Return VALUE as float when possible."""
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
