from dataclasses import dataclass
from typing import Protocol

import numpy as np

from config import VADSettings


class VADError(RuntimeError):
    """User-facing error for voice activity detector failures."""


class VoiceActivityDetector(Protocol):
    """Common interface for frame-level voice activity detectors."""

    @property
    def frame_samples(self) -> int:
        """Return how many int16 samples each VAD frame requires."""

    def is_speech(self, frame: np.ndarray, *, started: bool) -> bool:
        """Return whether FRAME contains speech."""

    def speech_probability(self, frame: np.ndarray) -> float:
        """Return the speech probability for FRAME."""


@dataclass
class TenVAD:
    """TEN VAD wrapper for frame-level speech probability detection."""

    settings: VADSettings

    def __post_init__(self) -> None:
        """Load the TEN VAD Python extension and initialize the detector."""
        try:
            from ten_vad_python import VAD
        except ImportError as exc:
            raise VADError(
                "TEN VAD is not installed. Build/install ten_vad_python for "
                "Raspberry Pi ARM64 before running voice mode."
            ) from exc

        self._vad = VAD(
            hop_size=self.settings.hop_size,
            threshold=self.settings.start_threshold,
        )

    @property
    def frame_samples(self) -> int:
        """Return the fixed TEN VAD hop size in samples."""
        return self.settings.hop_size

    def is_speech(self, frame: np.ndarray, *, started: bool) -> bool:
        """Classify one int16 audio frame using TEN VAD probability."""
        threshold = (
            self.settings.end_threshold if started else self.settings.start_threshold
        )
        return self.speech_probability(frame) >= threshold

    def speech_probability(self, frame: np.ndarray) -> float:
        """Return TEN VAD speech probability for one frame."""
        samples = _to_mono_int16(frame)
        if samples.size != self.settings.hop_size:
            raise VADError(
                "TEN VAD expected "
                f"{self.settings.hop_size} samples, got {samples.size}."
            )

        probability, _ = self._vad.process(samples)
        return float(probability)


def create_vad(settings: VADSettings) -> VoiceActivityDetector:
    """Create the project's voice activity detector."""
    return TenVAD(settings)


def _to_mono_int16(frame: np.ndarray) -> np.ndarray:
    """Return FRAME as a contiguous mono int16 array."""
    if frame.ndim == 2:
        frame = frame[:, 0]
    if frame.dtype != np.int16:
        frame = frame.astype(np.int16)
    return np.ascontiguousarray(frame.reshape(-1))
