from dataclasses import dataclass
from typing import Protocol

import numpy as np

from config import VADSettings


class VoiceActivityDetector(Protocol):
    """Common interface for frame-level voice activity detectors."""

    def is_speech(self, frame: np.ndarray, *, started: bool) -> bool:
        """Return whether FRAME contains speech."""


@dataclass(frozen=True)
class EnergyVAD:
    """A lightweight VAD based on normalized RMS energy.

    The detector uses a higher threshold before speech starts and a lower one
    after speech starts, which gives simple hysteresis against flicker.
    """

    settings: VADSettings

    def is_speech(self, frame: np.ndarray, *, started: bool) -> bool:
        """Classify one int16 audio frame using RMS energy."""
        threshold = (
            self.settings.end_threshold if started else self.settings.start_threshold
        )
        return _normalized_rms(frame) >= threshold


def _normalized_rms(frame: np.ndarray) -> float:
    """Return RMS energy normalized to roughly 0..1 for int16 audio."""
    if frame.size == 0:
        return 0.0

    samples = frame.astype(np.float32) / 32768.0
    return float(np.sqrt(np.mean(samples * samples)))
