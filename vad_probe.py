import argparse
import os
from statistics import mean

import numpy as np
import sounddevice as sd
from dotenv import load_dotenv

from config import (
    DEFAULT_RECORD_DEVICE,
    DEFAULT_RECORD_CHANNELS,
    DEFAULT_RECORD_SAMPLE_RATE,
    DEFAULT_VAD_FRAME_MS,
    DEFAULT_VAD_START_MS,
    DEFAULT_VAD_START_THRESHOLD,
)


def main() -> int:
    """Measure microphone RMS levels so VAD thresholds can be tuned."""
    load_dotenv()
    args = _parse_args()
    frame_samples = max(1, round(args.sample_rate * args.frame_ms / 1000))
    frame_count = max(1, round(args.duration * 1000 / args.frame_ms))
    start_frames = max(1, round(args.start_ms / args.frame_ms))

    print("Stay quiet until the measurement finishes.")
    print(
        "Config: "
        f"sample_rate={args.sample_rate}, channels={args.channels}, "
        f"device={args.device or 'system default'}, frame_ms={args.frame_ms}, "
        f"start_threshold={args.start_threshold}, start_ms={args.start_ms}"
    )

    try:
        levels = _record_levels(
            sample_rate=args.sample_rate,
            channels=args.channels,
            device=args.device,
            frame_samples=frame_samples,
            frame_count=frame_count,
        )
    except Exception as exc:
        print(f"VAD probe failed: {exc}")
        return 1

    if not levels:
        print("No audio frames were captured.")
        return 1

    stats = _build_stats(levels)
    longest_run = _longest_run_above(levels, args.start_threshold)
    suggested_threshold = max(
        args.start_threshold,
        stats["p99"] * 1.8,
        stats["max"] * 1.2,
    )

    print("\nBackground RMS:")
    print(
        f"mean={stats['mean']:.5f}, p90={stats['p90']:.5f}, "
        f"p95={stats['p95']:.5f}, p99={stats['p99']:.5f}, max={stats['max']:.5f}"
    )
    print(
        "Current trigger risk: "
        f"{longest_run} consecutive frames above threshold "
        f"({longest_run * args.frame_ms} ms)."
    )
    print(
        "Suggested VAD_START_THRESHOLD: "
        f"{suggested_threshold:.3f} to {suggested_threshold * 1.3:.3f}"
    )

    if longest_run >= start_frames:
        print("Result: current settings can trigger before speech.")
    else:
        print("Result: current settings should not trigger in this quiet sample.")
    return 0


def _parse_args() -> argparse.Namespace:
    """Return CLI arguments with .env-backed defaults."""
    parser = argparse.ArgumentParser(description="Measure microphone VAD energy.")
    parser.add_argument("--duration", type=float, default=5.0)
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=_read_positive_int("RECORD_SAMPLE_RATE", DEFAULT_RECORD_SAMPLE_RATE),
    )
    parser.add_argument(
        "--channels",
        type=int,
        default=_read_positive_int("RECORD_CHANNELS", DEFAULT_RECORD_CHANNELS),
    )
    parser.add_argument(
        "--device",
        default=_read_optional_env("RECORD_DEVICE", DEFAULT_RECORD_DEVICE),
    )
    parser.add_argument(
        "--frame-ms",
        type=int,
        default=_read_positive_int("VAD_FRAME_MS", DEFAULT_VAD_FRAME_MS),
    )
    parser.add_argument(
        "--start-threshold",
        type=float,
        default=_read_float("VAD_START_THRESHOLD", DEFAULT_VAD_START_THRESHOLD),
    )
    parser.add_argument(
        "--start-ms",
        type=int,
        default=_read_positive_int("VAD_START_MS", DEFAULT_VAD_START_MS),
    )
    return parser.parse_args()


def _record_levels(
    *,
    sample_rate: int,
    channels: int,
    device: str | None,
    frame_samples: int,
    frame_count: int,
) -> list[float]:
    """Record FRAME_COUNT frames and return normalized RMS levels."""
    levels: list[float] = []
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
            levels.append(_normalized_rms(frame))
    return levels


def _build_stats(levels: list[float]) -> dict[str, float]:
    """Return useful percentile statistics for LEVELS."""
    values = np.asarray(levels, dtype=np.float32)
    return {
        "mean": mean(levels),
        "p90": float(np.percentile(values, 90)),
        "p95": float(np.percentile(values, 95)),
        "p99": float(np.percentile(values, 99)),
        "max": float(np.max(values)),
    }


def _longest_run_above(levels: list[float], threshold: float) -> int:
    """Return the longest consecutive frame run above THRESHOLD."""
    longest = 0
    current = 0
    for level in levels:
        if level >= threshold:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return longest


def _normalized_rms(frame: np.ndarray) -> float:
    """Return RMS energy normalized to roughly 0..1 for int16 audio."""
    if frame.size == 0:
        return 0.0

    samples = frame.astype(np.float32) / 32768.0
    return float(np.sqrt(np.mean(samples * samples)))


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


def _read_float(name: str, default: float) -> float:
    """Return environment variable NAME as a float, or DEFAULT."""
    raw_value = os.getenv(name, "").strip()
    if not raw_value:
        return default

    try:
        return float(raw_value)
    except ValueError as exc:
        raise RuntimeError(f"{name} must be a number.") from exc


def _read_optional_env(name: str, default: str = "") -> str | None:
    """Return a non-empty environment value or None."""
    value = os.getenv(name, default).strip()
    return value or None


if __name__ == "__main__":
    raise SystemExit(main())
