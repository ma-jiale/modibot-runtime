from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from config import ASRSettings


class ASRError(RuntimeError):
    """User-facing error for speech recognition failures."""


@dataclass(frozen=True)
class TranscriptionResult:
    """The recognized text and lightweight metadata from an ASR request."""

    text: str
    language: str | None
    duration: float | None


class SpeechRecognizer(Protocol):
    """Common interface implemented by concrete ASR providers."""

    def transcribe(self, audio_path: Path) -> TranscriptionResult:
        """Return recognized text from AUDIO_PATH."""


class FasterWhisperASR:
    """Local ASR provider backed by faster-whisper."""

    def __init__(self, settings: ASRSettings) -> None:
        """Store SETTINGS and lazily load the model on first use."""
        self._settings = settings
        self._model = None

    def transcribe(self, audio_path: Path) -> TranscriptionResult:
        """Transcribe AUDIO_PATH with faster-whisper."""
        if not audio_path.exists():
            raise ASRError(f"Audio file does not exist: {audio_path}")

        model = self._load_model()
        try:
            segments, info = model.transcribe(
                str(audio_path),
                language=self._settings.language,
                vad_filter=True,
            )
            text = "".join(segment.text for segment in segments).strip()
        except Exception as exc:
            raise ASRError(f"ASR transcription failed: {exc}") from exc

        if not text:
            raise ASRError("ASR returned empty text.")

        return TranscriptionResult(
            text=text,
            language=getattr(info, "language", None),
            duration=getattr(info, "duration", None),
        )

    def _load_model(self):
        """Load the Whisper model once and reuse it for later turns."""
        if self._model is not None:
            return self._model

        try:
            from faster_whisper import WhisperModel

            self._model = WhisperModel(
                self._settings.model_size,
                device=self._settings.device,
                compute_type=self._settings.compute_type,
            )
        except Exception as exc:
            raise ASRError(f"Failed to load faster-whisper model: {exc}") from exc

        return self._model


def create_speech_recognizer(settings: ASRSettings) -> SpeechRecognizer:
    """Create a concrete ASR provider from SETTINGS."""
    if settings.provider == "faster-whisper":
        return FasterWhisperASR(settings)
    raise ASRError(f"Unsupported ASR provider: {settings.provider}")
