# Voice Conversation Agent - Text Prototype

This project is a command-line chat agent. It uses an OpenAI-compatible chat API for text conversation and Windows system TTS for speech output.

## Install

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Chat Configuration

Copy the example config:

```powershell
Copy-Item .env.example .env
```

Edit `.env`:

```env
MINIMAX_API_KEY=your_minimax_api_key_here
MINIMAX_BASE_URL=https://api.minimaxi.com/v1
MINIMAX_MODEL=MiniMax-M2.7
MINIMAX_API_MODE=chat
MAX_HISTORY_TURNS=20
```

If you override `AGENT_SYSTEM_PROMPT`, keep the Simplified Chinese requirement:

```env
AGENT_SYSTEM_PROMPT=Reply in concise Simplified Chinese. Do not use Traditional Chinese unless explicitly requested.
```

## Run

```powershell
python main.py
```

Commands:

- `voice` / `v`: record one utterance, transcribe it, and send it to the agent
- `exit` / `quit`: stop the program
- `reset` / `clear`: clear the current conversation

## Voice Input

The ASR module uses a generic recognizer interface. The first provider is
`faster-whisper`, which runs locally and does not need an ASR API key.

The `voice` command enters a continuous voice loop. The recorder listens for
speech, stops automatically after sustained silence, transcribes the saved WAV,
sends the text to the agent, plays the answer, and then starts listening again.
Say `退出语音模式` to leave voice mode.

```env
ASR_PROVIDER=faster-whisper
ASR_MODEL_SIZE=base
ASR_DEVICE=cpu
ASR_COMPUTE_TYPE=int8
ASR_LANGUAGE=zh
RECORDINGS_DIR=recordings
RECORD_SAMPLE_RATE=16000
RECORD_CHANNELS=1
VAD_FRAME_MS=30
VAD_START_THRESHOLD=0.018
VAD_END_THRESHOLD=0.012
VAD_SILENCE_MS=900
VAD_MIN_SPEECH_MS=300
VAD_MAX_RECORD_SECONDS=20
VAD_PREROLL_MS=300
```

Notes:

- `ASR_MODEL_SIZE=tiny` is faster but less accurate.
- `ASR_MODEL_SIZE=base` is a good CPU MVP default.
- The first run downloads the model.
- Recordings are saved in `recordings/`, which is ignored by git.
- If recording fails on Windows, try `RECORD_SAMPLE_RATE=44100` to match your microphone device.
- Raise `VAD_START_THRESHOLD` if background noise starts recordings too easily.
- Lower `VAD_START_THRESHOLD` if speech is not detected.

## Text To Speech

The TTS module uses the local `system` provider, which relies on Windows built-in speech synthesis and does not need a cloud TTS plan.

Enable local Windows TTS in `.env`:

```env
TTS_ENABLED=true
TTS_PROVIDER=system
TTS_FORMAT=wav
TTS_OUTPUT_DIR=outputs
TTS_AUTOPLAY=true
TTS_STREAMING=true
TTS_STREAM_CHUNK_CHARS=60
```

Notes:

- `system` outputs WAV files and works only on Windows for now.
- Audio files are saved in `TTS_OUTPUT_DIR`, which defaults to `outputs`.
- `outputs/` is ignored by git.
- `TTS_STREAMING=true` makes the agent speak short sentence chunks while the model is still generating.
- `TTS_STREAM_CHUNK_CHARS` controls the maximum chunk size when punctuation is sparse.
- In streaming mode, chunks are played immediately regardless of `TTS_AUTOPLAY`; that flag only affects non-streaming playback.

## Files

- `main.py`: command-line chat loop
- `agent.py`: OpenAI-compatible chat API calls and error handling
- `conversation.py`: bounded conversation history
- `config.py`: environment variable loading and validation
- `recorder.py`: microphone recording and WAV saving
- `asr.py`: generic ASR interface plus faster-whisper provider
- `voice_activity.py`: energy-based voice activity detection
- `tts.py`: generic TTS interface plus the Windows system provider
- `streaming_tts.py`: streaming text chunking, TTS byte generation, and ordered audio playback
- `.env.example`: local configuration template; never commit real `.env` values
