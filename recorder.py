import wave
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import sounddevice as sd

from config import ASRSettings


class RecorderError(RuntimeError):
    """User-facing error for microphone recording failures."""


@dataclass(frozen=True)
class RecordingResult:
    """The path and audio properties of one saved recording."""

    path: Path
    sample_rate: int
    channels: int


class WavRecorder:
    """Record microphone audio and save it as PCM WAV."""

    def __init__(self, settings: ASRSettings) -> None:
        """Store recorder settings from ASRSettings."""
        self._settings = settings

    def record_until_enter(self) -> RecordingResult:
        """Record from the microphone until the user presses Enter."""
        output_path = self._build_output_path()
        frames: list[np.ndarray] = []

        def callback(indata, frames_count, time_info, status) -> None:
            """Collect each audio block from sounddevice."""
            if status:
                print(f"Recorder status: {status}")
            frames.append(indata.copy())

        try:
            with sd.InputStream(
                samplerate=self._settings.sample_rate,
                channels=self._settings.channels,
                dtype="int16",
                callback=callback,
            ):
                input("Recording... press Enter to stop.")
        except Exception as exc:
            raise RecorderError(f"Recording failed: {exc}") from exc

        if not frames:
            raise RecorderError("No audio was recorded.")

        audio = np.concatenate(frames, axis=0)
        self._write_wav(output_path, audio)
        return RecordingResult(
            path=output_path,
            sample_rate=self._settings.sample_rate,
            channels=self._settings.channels,
        )

    def _build_output_path(self) -> Path:
        """Return a unique recording path."""
        output_dir = Path(self._settings.recordings_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        return output_dir / f"input_{timestamp}.wav"

    def _write_wav(self, path: Path, audio: np.ndarray) -> None:
        """Write int16 AUDIO to PATH as mono/stereo PCM WAV."""
        with wave.open(str(path), "wb") as wav_file:
            wav_file.setnchannels(self._settings.channels)
            wav_file.setsampwidth(2)
            wav_file.setframerate(self._settings.sample_rate)
            wav_file.writeframes(audio.tobytes())
