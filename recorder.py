import wave
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import sounddevice as sd

from config import ASRSettings
from voice_activity import VoiceActivityDetector


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

    def record_until_silence(self, vad: VoiceActivityDetector) -> RecordingResult:
        """Record one utterance, ending after sustained silence."""
        frame_samples = _frame_samples(
            self._settings.sample_rate, self._settings.vad.frame_ms
        )
        silence_frames = _frames_for_ms(
            self._settings.vad.silence_ms, self._settings.vad.frame_ms
        )
        min_speech_frames = _frames_for_ms(
            self._settings.vad.min_speech_ms, self._settings.vad.frame_ms
        )
        max_frames = max(
            1,
            round(
                self._settings.vad.max_record_seconds
                * 1000
                / self._settings.vad.frame_ms
            ),
        )
        preroll_frames = _frames_for_ms(
            self._settings.vad.preroll_ms, self._settings.vad.frame_ms
        )

        output_path = self._build_output_path()
        started = False
        speech_frames = 0
        quiet_frames = 0
        captured: list[np.ndarray] = []
        preroll: deque[np.ndarray] = deque(maxlen=preroll_frames)

        print("Listening... speak now.")
        try:
            with sd.InputStream(
                samplerate=self._settings.sample_rate,
                channels=self._settings.channels,
                dtype="int16",
                blocksize=frame_samples,
            ) as stream:
                for _ in range(max_frames):
                    frame, overflowed = stream.read(frame_samples)
                    if overflowed:
                        print("Recorder warning: input overflow")

                    is_speech = vad.is_speech(frame, started=started)
                    if not started:
                        preroll.append(frame.copy())
                        if is_speech:
                            print("Speech detected.")
                            started = True
                            captured.extend(preroll)
                            preroll.clear()
                        continue

                    captured.append(frame.copy())
                    if is_speech:
                        speech_frames += 1
                        quiet_frames = 0
                    else:
                        quiet_frames += 1
                        if (
                            speech_frames >= min_speech_frames
                            and quiet_frames >= silence_frames
                        ):
                            break
        except Exception as exc:
            raise RecorderError(f"Recording failed: {exc}") from exc

        if not captured:
            raise RecorderError("No speech was detected.")

        audio = np.concatenate(captured, axis=0)
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


def _frame_samples(sample_rate: int, frame_ms: int) -> int:
    """Return the number of samples in one VAD frame."""
    return max(1, round(sample_rate * frame_ms / 1000))


def _frames_for_ms(duration_ms: int, frame_ms: int) -> int:
    """Return how many VAD frames cover DURATION_MS."""
    return max(1, round(duration_ms / frame_ms))
