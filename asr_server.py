import os
import tempfile
from pathlib import Path
from typing import Annotated

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, UploadFile

from config import ServerASRSettings, load_server_asr_settings
from server_asr import FasterWhisperASR


DEFAULT_SERVER_MAX_UPLOAD_MB = 32

app = FastAPI(title="Voice Agent ASR Server")
_recognizer: FasterWhisperASR | None = None


def get_recognizer() -> FasterWhisperASR:
    """Return the process-wide ASR model, loading it on first use."""
    global _recognizer
    if _recognizer is None:
        _recognizer = FasterWhisperASR(_read_server_settings())
    return _recognizer


@app.get("/health")
def health() -> dict[str, str]:
    """Return a lightweight readiness response."""
    return {"status": "ok"}


@app.post("/v1/transcriptions")
async def transcribe(
    file: Annotated[UploadFile, File()],
    language: Annotated[str | None, Form()] = None,
    authorization: Annotated[str | None, Header()] = None,
    recognizer: FasterWhisperASR = Depends(get_recognizer),
) -> dict[str, str | float | None]:
    """Transcribe one uploaded audio file and return JSON text metadata."""
    _check_authorization(authorization)
    _check_upload(file)

    suffix = Path(file.filename or "input.wav").suffix or ".wav"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as temp_file:
        temp_path = Path(temp_file.name)

    try:
        size = await _save_upload(file, temp_path)
        _check_upload_size(size)

        result = recognizer.transcribe_with_language(temp_path, language)
    finally:
        try:
            temp_path.unlink(missing_ok=True)
        except OSError:
            pass

    return {
        "text": result.text,
        "language": language or result.language,
        "duration": result.duration,
    }


def _read_server_settings() -> ServerASRSettings:
    """Return faster-whisper settings for the ASR server process."""
    return load_server_asr_settings()


def _check_authorization(authorization: str | None) -> None:
    """Reject requests when ASR_SERVER_API_KEY is set and the token is wrong."""
    expected = os.getenv("ASR_SERVER_API_KEY", "").strip()
    if not expected:
        return

    if authorization != f"Bearer {expected}":
        raise HTTPException(status_code=401, detail="Invalid ASR server token.")


def _check_upload(file: UploadFile) -> None:
    """Reject requests without an uploaded audio file."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="Missing audio file.")


async def _save_upload(file: UploadFile, path: Path) -> int:
    """Write uploaded FILE to PATH and return its byte size."""
    size = 0
    with path.open("wb") as output:
        while True:
            chunk = await file.read(1024 * 1024)
            if not chunk:
                return size
            size += len(chunk)
            _check_upload_size(size)
            output.write(chunk)


def _check_upload_size(size: int) -> None:
    """Reject uploads larger than ASR_SERVER_MAX_UPLOAD_MB."""
    limit_mb = _read_positive_int(
        "ASR_SERVER_MAX_UPLOAD_MB", DEFAULT_SERVER_MAX_UPLOAD_MB
    )
    limit_bytes = limit_mb * 1024 * 1024
    if size > limit_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"Audio upload exceeds {limit_mb} MB.",
        )


def _read_positive_int(name: str, default: int) -> int:
    """Return environment variable NAME as a positive int, or DEFAULT."""
    raw_value = os.getenv(name, "").strip()
    if not raw_value:
        return default

    try:
        value = int(raw_value)
    except ValueError as exc:
        raise RuntimeError(f"{name} must be a positive integer.") from exc

    if value <= 0:
        raise RuntimeError(f"{name} must be greater than 0.")
    return value
