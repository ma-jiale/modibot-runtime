import argparse
from statistics import mean

import numpy as np
import sounddevice as sd

from config import load_asr_settings
from voice_activity import VADError, create_vad


def main() -> int:
    """Measure TEN VAD speech probabilities for threshold tuning."""
    try:
        settings = load_asr_settings()
        vad = create_vad(settings.vad)
    except (RuntimeError, VADError) as exc:
        print(f"VAD probe failed: {exc}")
        return 1

    args = _parse_args()
    frame_ms = vad.frame_samples * 1000 / settings.sample_rate
    frame_count = max(1, round(args.duration * 1000 / frame_ms))
    start_frames = max(1, round(settings.vad.start_ms / frame_ms))

    print("Stay quiet until the measurement finishes.")
    print(
        "Config: "
        f"sample_rate={settings.sample_rate}, channels={settings.channels}, "
        f"device={settings.record_device or 'system default'}, "
        f"hop_size={vad.frame_samples}, "
        f"start_threshold={settings.vad.start_threshold}, "
        f"start_ms={settings.vad.start_ms}"
    )

    try:
        probabilities = _record_probabilities(
            sample_rate=settings.sample_rate,
            channels=settings.channels,
            device=settings.record_device,
            frame_samples=vad.frame_samples,
            frame_count=frame_count,
            vad=vad,
        )
    except Exception as exc:
        print(f"VAD probe failed: {exc}")
        return 1

    if not probabilities:
        print("No audio frames were captured.")
        return 1

    stats = _build_stats(probabilities)
    longest_run = _longest_run_above(
        probabilities, settings.vad.start_threshold
    )
    suggested_threshold = min(
        0.95,
        max(settings.vad.start_threshold, stats["p99"] + 0.10, stats["max"] + 0.05),
    )

    print("\nTEN VAD speech probability while quiet:")
    print(
        f"mean={stats['mean']:.3f}, p90={stats['p90']:.3f}, "
        f"p95={stats['p95']:.3f}, p99={stats['p99']:.3f}, max={stats['max']:.3f}"
    )
    print(
        "Current trigger risk: "
        f"{longest_run} consecutive frames above threshold "
        f"({longest_run * frame_ms:.0f} ms)."
    )
    print(
        "Suggested TEN_VAD_START_THRESHOLD: "
        f"{suggested_threshold:.2f} to {min(0.99, suggested_threshold + 0.10):.2f}"
    )

    if longest_run >= start_frames:
        print("Result: current TEN VAD settings can trigger before speech.")
    else:
        print("Result: current TEN VAD settings should not trigger in this sample.")
    return 0


def _parse_args() -> argparse.Namespace:
    """Return CLI arguments."""
    parser = argparse.ArgumentParser(description="Measure TEN VAD probabilities.")
    parser.add_argument("--duration", type=float, default=8.0)
    return parser.parse_args()


def _record_probabilities(
    *,
    sample_rate: int,
    channels: int,
    device: str | None,
    frame_samples: int,
    frame_count: int,
    vad,
) -> list[float]:
    """Record FRAME_COUNT frames and return TEN VAD speech probabilities."""
    probabilities: list[float] = []
    with sd.InputStream(
        samplerate=sample_rate,
        channels=channels,
        dtype="int16",
        blocksize=frame_samples,
        device=device,
    ) as stream:
        for _ in range(frame_count):
            frame, overflowed = stream.read(frame_samples)
            if overflowed:
                print("Recorder warning: input overflow")
            probabilities.append(vad.speech_probability(frame))
    return probabilities


def _build_stats(values: list[float]) -> dict[str, float]:
    """Return useful percentile statistics for VALUES."""
    samples = np.asarray(values, dtype=np.float32)
    return {
        "mean": mean(values),
        "p90": float(np.percentile(samples, 90)),
        "p95": float(np.percentile(samples, 95)),
        "p99": float(np.percentile(samples, 99)),
        "max": float(np.max(samples)),
    }


def _longest_run_above(values: list[float], threshold: float) -> int:
    """Return the longest consecutive frame run above THRESHOLD."""
    longest = 0
    current = 0
    for value in values:
        if value >= threshold:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return longest


if __name__ == "__main__":
    raise SystemExit(main())
