from pathlib import Path

import sounddevice as sd

from config import load_asr_settings
from recorder import RecorderError, WavRecorder
from voice_activity import EnergyVAD


def main() -> int:
    """Print audio devices and record one short VAD-gated test file."""
    try:
        settings = load_asr_settings()
    except RuntimeError as exc:
        print(f"Config error: {exc}")
        return 1

    print("Audio devices visible to sounddevice:")
    print(sd.query_devices())
    print()
    print(f"Configured microphone: {settings.record_device or 'system default'}")
    print(f"Sample rate: {settings.sample_rate}")
    print(f"Channels: {settings.channels}")
    print()
    print("Speak after the listening prompt. The test stops after silence.")

    recorder = WavRecorder(settings)
    vad = EnergyVAD(settings.vad)
    try:
        recording = recorder.record_until_silence(vad)
    except RecorderError as exc:
        print(f"Recording failed: {exc}")
        return 1

    print(f"Saved test recording: {Path(recording.path)}")
    print("On Raspberry Pi, play it with: aplay <path>")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
