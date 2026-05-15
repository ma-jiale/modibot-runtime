from pathlib import Path

from asr import ASRError, TranscriptionResult
from config import ServerASRSettings


class FasterWhisperASR:
    """Server-side ASR provider backed by faster-whisper."""

    def __init__(self, settings: ServerASRSettings) -> None:
        """Store SETTINGS and lazily load the Whisper model."""
        self._settings = settings
        self._model = None

    def transcribe_with_language(
        self, audio_path: Path, language: str | None
    ) -> TranscriptionResult:
        """Transcribe AUDIO_PATH with an optional per-request LANGUAGE."""
        if not audio_path.exists():
            raise ASRError(f"Audio file does not exist: {audio_path}")

        model = self._load_model()
        try:
            segments, info = model.transcribe(
                str(audio_path),
                language=language,
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
        """Load the Whisper model once and reuse it for later requests."""
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
