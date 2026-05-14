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

## Run

```powershell
python main.py
```

Commands:

- `exit` / `quit`: stop the program
- `reset` / `clear`: clear the current conversation

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
- `tts.py`: generic TTS interface plus the Windows system provider
- `streaming_tts.py`: streaming text chunking, TTS byte generation, and ordered audio playback
- `.env.example`: local configuration template; never commit real `.env` values
