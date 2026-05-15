from dataclasses import dataclass
import os

from dotenv import load_dotenv


DEFAULT_PROVIDER = "minimax"
DEFAULT_MODEL = "MiniMax-M2.7"
DEFAULT_BASE_URL = "https://api.minimaxi.com/v1"
DEFAULT_MAX_HISTORY_TURNS = 20
DEFAULT_API_MODE = "chat"
DEFAULT_ASR_PROVIDER = "remote"
DEFAULT_ASR_LANGUAGE = "zh"
DEFAULT_ASR_REMOTE_TIMEOUT = 60.0
DEFAULT_RECORD_DEVICE = "seeed2micvoicec"
DEFAULT_RECORDINGS_DIR = "recordings"
DEFAULT_RECORD_SAMPLE_RATE = 16000
DEFAULT_RECORD_CHANNELS = 1
DEFAULT_SERVER_ASR_MODEL_SIZE = "medium"
DEFAULT_SERVER_ASR_DEVICE = "cuda"
DEFAULT_SERVER_ASR_COMPUTE_TYPE = "float16"
DEFAULT_TEN_VAD_HOP_SIZE = 256
DEFAULT_TEN_VAD_START_THRESHOLD = 0.5
DEFAULT_TEN_VAD_END_THRESHOLD = 0.35
DEFAULT_VAD_START_MS = 300
DEFAULT_VAD_SILENCE_MS = 900
DEFAULT_VAD_MIN_SPEECH_MS = 300
DEFAULT_VAD_MAX_RECORD_SECONDS = 20.0
DEFAULT_VAD_PREROLL_MS = 300
DEFAULT_TTS_PROVIDER = "none"
DEFAULT_DOUBAO_TTS_ENDPOINT = "wss://openspeech.bytedance.com/api/v3/tts/bidirection"
DEFAULT_DOUBAO_TTS_RESOURCE_ID = "seed-tts-2.0"
DEFAULT_DOUBAO_TTS_SPEAKER = "zh_female_cancan_mars_bigtts"
DEFAULT_TTS_USER_UID = "voice-agent"
DEFAULT_TTS_AUDIO_FORMAT = "pcm"
DEFAULT_TTS_SAMPLE_RATE = 24000
DEFAULT_TTS_CHANNELS = 1
DEFAULT_TTS_SPEECH_RATE = 0
DEFAULT_TTS_LOUDNESS_RATE = 0
DEFAULT_TTS_CONNECT_TIMEOUT = 10.0
DEFAULT_TTS_SESSION_TIMEOUT = 120.0
DEFAULT_SYSTEM_PROMPT = (
    "You are the text prototype of a voice conversation agent. "
    "Reply in natural, concise Simplified Chinese that is suitable for being read aloud. "
    "Do not use Traditional Chinese unless the user explicitly asks for it. "
    "If the user's request is unclear, ask one short clarifying question first."
)
VALID_API_MODES = {"chat", "responses"}
VALID_TTS_PROVIDERS = {"none", "doubao"}
VALID_TTS_AUDIO_FORMATS = {"pcm"}


@dataclass(frozen=True)
class Settings:
    """Runtime configuration loaded from environment variables."""

    provider: str
    api_key: str
    model: str
    system_prompt: str
    base_url: str | None
    api_mode: str
    default_headers: dict[str, str]
    max_history_turns: int
    asr: "ASRSettings"
    tts: "TTSSettings"


@dataclass(frozen=True)
class ASRSettings:
    """Runtime configuration for speech-to-text and recording."""

    provider: str
    language: str | None
    remote_url: str | None
    remote_timeout: float
    remote_api_key: str | None
    record_device: str | None
    recordings_dir: str
    sample_rate: int
    channels: int
    vad: "VADSettings"


@dataclass(frozen=True)
class ServerASRSettings:
    """Runtime configuration for the GPU ASR server."""

    model_size: str
    device: str
    compute_type: str
    language: str | None


@dataclass(frozen=True)
class VADSettings:
    """Runtime configuration for TEN voice activity detection."""

    hop_size: int
    start_threshold: float
    end_threshold: float
    start_ms: int
    silence_ms: int
    min_speech_ms: int
    max_record_seconds: float
    preroll_ms: int


@dataclass(frozen=True)
class TTSSettings:
    """Runtime configuration for assistant speech playback."""

    provider: str
    api_key: str | None
    endpoint: str
    resource_id: str
    speaker: str
    user_uid: str
    audio_format: str
    sample_rate: int
    channels: int
    output_device: str | None
    speech_rate: int
    loudness_rate: int
    connect_timeout: float
    session_timeout: float


def load_settings() -> Settings:
    """Load and validate app settings from .env and the process environment.

    MiniMax-specific variables are preferred, while OPENAI_* names remain
    supported so the same code can target other OpenAI-compatible providers.
    """
    load_dotenv()

    provider = os.getenv("AI_PROVIDER", DEFAULT_PROVIDER).strip().lower()
    api_key = _read_first_env("MINIMAX_API_KEY", "OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "API key is missing. Set MINIMAX_API_KEY or copy .env.example to .env."
        )

    model = _read_first_env("MINIMAX_MODEL", "OPENAI_MODEL") or DEFAULT_MODEL
    system_prompt = (
        os.getenv("AGENT_SYSTEM_PROMPT", DEFAULT_SYSTEM_PROMPT).strip()
        or DEFAULT_SYSTEM_PROMPT
    )

    return Settings(
        provider=provider,
        api_key=api_key,
        model=model,
        system_prompt=system_prompt,
        base_url=_read_base_url(),
        api_mode=_read_api_mode(),
        default_headers=_read_default_headers(),
        max_history_turns=_read_positive_int(
            "MAX_HISTORY_TURNS", DEFAULT_MAX_HISTORY_TURNS
        ),
        asr=_read_asr_settings(),
        tts=_read_tts_settings(),
    )


def load_asr_settings() -> ASRSettings:
    """Load only ASR and recording settings from the environment."""
    load_dotenv()
    return _read_asr_settings()


def load_server_asr_settings() -> ServerASRSettings:
    """Load only server-side faster-whisper settings."""
    load_dotenv()
    return ServerASRSettings(
        model_size=_read_first_env(
            "ASR_MODEL_SIZE", default=DEFAULT_SERVER_ASR_MODEL_SIZE
        ),
        device=_normalize_asr_device(
            _read_first_env("ASR_DEVICE", default=DEFAULT_SERVER_ASR_DEVICE)
        ),
        compute_type=_read_first_env(
            "ASR_COMPUTE_TYPE", default=DEFAULT_SERVER_ASR_COMPUTE_TYPE
        ),
        language=_read_first_env("ASR_LANGUAGE", default=DEFAULT_ASR_LANGUAGE)
        or None,
    )


def _read_asr_settings() -> ASRSettings:
    """Load speech-to-text and recorder settings."""
    language = _read_first_env("ASR_LANGUAGE", default=DEFAULT_ASR_LANGUAGE)
    remote_url = _read_first_env("ASR_REMOTE_URL") or None
    remote_api_key = _read_first_env("ASR_REMOTE_API_KEY") or None
    return ASRSettings(
        provider=_read_first_env("ASR_PROVIDER", default=DEFAULT_ASR_PROVIDER).lower(),
        language=language or None,
        remote_url=remote_url,
        remote_timeout=_read_positive_float(
            "ASR_REMOTE_TIMEOUT", DEFAULT_ASR_REMOTE_TIMEOUT
        ),
        remote_api_key=remote_api_key,
        record_device=_read_optional_env("RECORD_DEVICE", DEFAULT_RECORD_DEVICE),
        recordings_dir=_read_first_env(
            "RECORDINGS_DIR", default=DEFAULT_RECORDINGS_DIR
        ),
        sample_rate=_read_positive_int(
            "RECORD_SAMPLE_RATE", DEFAULT_RECORD_SAMPLE_RATE
        ),
        channels=_read_positive_int("RECORD_CHANNELS", DEFAULT_RECORD_CHANNELS),
        vad=_read_vad_settings(),
    )


def _read_vad_settings() -> VADSettings:
    """Load TEN VAD settings."""
    return VADSettings(
        hop_size=_read_valid_ten_vad_hop_size(),
        start_threshold=_read_probability(
            "TEN_VAD_START_THRESHOLD", DEFAULT_TEN_VAD_START_THRESHOLD
        ),
        end_threshold=_read_probability(
            "TEN_VAD_END_THRESHOLD", DEFAULT_TEN_VAD_END_THRESHOLD
        ),
        start_ms=_read_positive_int("VAD_START_MS", DEFAULT_VAD_START_MS),
        silence_ms=_read_positive_int("VAD_SILENCE_MS", DEFAULT_VAD_SILENCE_MS),
        min_speech_ms=_read_positive_int(
            "VAD_MIN_SPEECH_MS", DEFAULT_VAD_MIN_SPEECH_MS
        ),
        max_record_seconds=_read_float(
            "VAD_MAX_RECORD_SECONDS", DEFAULT_VAD_MAX_RECORD_SECONDS
        ),
        preroll_ms=_read_positive_int("VAD_PREROLL_MS", DEFAULT_VAD_PREROLL_MS),
    )


def _read_tts_settings() -> TTSSettings:
    """Load text-to-speech settings."""
    provider = _normalize_tts_provider(
        _read_first_env("TTS_PROVIDER", default=DEFAULT_TTS_PROVIDER)
    )
    if provider not in VALID_TTS_PROVIDERS:
        valid_providers = ", ".join(sorted(VALID_TTS_PROVIDERS))
        raise RuntimeError(f"TTS_PROVIDER must be one of: {valid_providers}.")

    audio_format = _read_first_env(
        "TTS_AUDIO_FORMAT", default=DEFAULT_TTS_AUDIO_FORMAT
    ).lower()
    if audio_format not in VALID_TTS_AUDIO_FORMATS:
        valid_formats = ", ".join(sorted(VALID_TTS_AUDIO_FORMATS))
        raise RuntimeError(f"TTS_AUDIO_FORMAT must be one of: {valid_formats}.")

    api_key = _read_first_env("DOUBAO_TTS_API_KEY", "VOLCENGINE_TTS_API_KEY") or None
    if provider == "doubao" and not api_key:
        raise RuntimeError("DOUBAO_TTS_API_KEY is required when TTS_PROVIDER=doubao.")

    return TTSSettings(
        provider=provider,
        api_key=api_key,
        endpoint=_read_first_env(
            "DOUBAO_TTS_ENDPOINT", default=DEFAULT_DOUBAO_TTS_ENDPOINT
        ),
        resource_id=_read_first_env(
            "DOUBAO_TTS_RESOURCE_ID", default=DEFAULT_DOUBAO_TTS_RESOURCE_ID
        ),
        speaker=_read_first_env(
            "DOUBAO_TTS_SPEAKER", default=DEFAULT_DOUBAO_TTS_SPEAKER
        ),
        user_uid=_read_first_env("TTS_USER_UID", default=DEFAULT_TTS_USER_UID),
        audio_format=audio_format,
        sample_rate=_read_positive_int("TTS_SAMPLE_RATE", DEFAULT_TTS_SAMPLE_RATE),
        channels=_read_positive_int("TTS_CHANNELS", DEFAULT_TTS_CHANNELS),
        output_device=_read_optional_env("TTS_OUTPUT_DEVICE"),
        speech_rate=_read_int_in_range(
            "TTS_SPEECH_RATE", DEFAULT_TTS_SPEECH_RATE, -50, 100
        ),
        loudness_rate=_read_int_in_range(
            "TTS_LOUDNESS_RATE", DEFAULT_TTS_LOUDNESS_RATE, -50, 100
        ),
        connect_timeout=_read_positive_float(
            "TTS_CONNECT_TIMEOUT", DEFAULT_TTS_CONNECT_TIMEOUT
        ),
        session_timeout=_read_positive_float(
            "TTS_SESSION_TIMEOUT", DEFAULT_TTS_SESSION_TIMEOUT
        ),
    )


def _normalize_tts_provider(provider: str) -> str:
    """Return the canonical TTS provider name for user-facing aliases."""
    normalized = provider.strip().lower()
    if normalized in {"", "off", "disabled", "system"}:
        return "none"
    return normalized


def _read_base_url() -> str | None:
    """Return the normalized OpenAI-compatible API base URL."""
    raw_url = _read_first_env(
        "MINIMAX_BASE_URL",
        "OPENAI_BASE_URL",
        "OPENAI_API_BASE",
        default=DEFAULT_BASE_URL,
    )
    return _normalize_base_url(raw_url)


def _normalize_asr_device(device: str) -> str:
    """Return the faster-whisper device name for common user aliases."""
    normalized = device.strip().lower()
    if normalized == "gpu":
        return "cuda"
    return normalized


def _read_api_mode() -> str:
    """Return the selected API mode after validating it."""
    api_mode = (
        _read_first_env("MINIMAX_API_MODE", "OPENAI_API_MODE", default=DEFAULT_API_MODE)
        .strip()
        .lower()
    )
    if api_mode not in VALID_API_MODES:
        valid_modes = ", ".join(sorted(VALID_API_MODES))
        raise RuntimeError(f"MINIMAX_API_MODE must be one of: {valid_modes}.")
    return api_mode


def _read_default_headers() -> dict[str, str]:
    """Return optional provider headers, omitting unset values."""
    headers = {
        "User-Agent": _read_first_env("MINIMAX_USER_AGENT", "OPENAI_USER_AGENT"),
        "HTTP-Referer": _read_first_env(
            "MINIMAX_HTTP_REFERER", "OPENAI_HTTP_REFERER"
        ),
        "X-Title": _read_first_env("MINIMAX_X_TITLE", "OPENAI_X_TITLE"),
    }
    return {key: value for key, value in headers.items() if value}


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


def _read_probability(name: str, default: float) -> float:
    """Return environment variable NAME as a probability in the range 0..1."""
    value = _read_float(name, default)
    if not 0 <= value <= 1:
        raise RuntimeError(f"{name} must be between 0 and 1.")
    return value


def _read_int_in_range(name: str, default: int, minimum: int, maximum: int) -> int:
    """Return environment variable NAME as an int within MINIMUM..MAXIMUM."""
    raw_value = os.getenv(name, "").strip()
    if not raw_value:
        return default

    try:
        value = int(raw_value)
    except ValueError as exc:
        raise RuntimeError(f"{name} must be an integer.") from exc

    if not minimum <= value <= maximum:
        raise RuntimeError(f"{name} must be between {minimum} and {maximum}.")
    return value


def _read_positive_float(name: str, default: float) -> float:
    """Return environment variable NAME as a positive float, or DEFAULT."""
    value = _read_float(name, default)
    if value <= 0:
        raise RuntimeError(f"{name} must be greater than 0.")
    return value


def _read_valid_ten_vad_hop_size() -> int:
    """Return a supported TEN VAD hop size."""
    hop_size = _read_positive_int("TEN_VAD_HOP_SIZE", DEFAULT_TEN_VAD_HOP_SIZE)
    if hop_size not in {160, 256}:
        raise RuntimeError("TEN_VAD_HOP_SIZE must be 160 or 256.")
    return hop_size


def _normalize_base_url(raw_url: str) -> str | None:
    """Normalize RAW_URL to the API root expected by the OpenAI SDK.

    Users sometimes paste a full endpoint path. The SDK appends endpoint paths
    itself, so this function trims known suffixes like /chat/completions.
    """
    if not raw_url:
        return None

    base_url = raw_url.rstrip("/")
    for suffix in ("/chat/completions", "/responses"):
        if base_url.endswith(suffix):
            base_url = base_url[: -len(suffix)]

    return base_url


def _read_first_env(*names: str, default: str = "") -> str:
    """Return the first non-empty environment value among NAMES."""
    for name in names:
        value = os.getenv(name, "").strip()
        if value:
            return value
    return default.strip()


def _read_optional_env(name: str, default: str = "") -> str | None:
    """Return a non-empty environment value or None."""
    value = _read_first_env(name, default=default)
    return value or None
