# Raspberry Pi Voice Agent

This project is a Raspberry Pi command-line voice agent. The Pi records speech
from a ReSpeaker 2-Mics Pi HAT, uses TEN VAD for local voice turn detection,
sends the WAV file to a LAN ASR server, sends the recognized text to an
OpenAI-compatible chat API, and prints the assistant reply in the terminal.

TTS is intentionally not implemented in this client yet.

## Raspberry Pi Setup

Install system audio dependencies:

```bash
sudo apt update
sudo apt install -y python3-venv portaudio19-dev alsa-utils
```

Create the Python environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Copy and edit the config:

```bash
cp .env.example .env
nano .env
```

Set your API key and ASR server address:

```env
MINIMAX_API_KEY=your_minimax_api_key_here
ASR_REMOTE_URL=http://192.168.1.100:8000/v1/transcriptions
```

If the Raspberry Pi uses an HTTP or SOCKS proxy for external APIs, exclude the
LAN ASR server from the proxy:

```bash
export NO_PROXY=192.168.1.100,localhost,127.0.0.1
export no_proxy=192.168.1.100,localhost,127.0.0.1
```

Add those two lines to `~/.bashrc` if the ASR server is a fixed LAN host.

## TEN VAD

This client uses TEN VAD as the only voice activity detector. Build and install
the TEN VAD Python extension on Raspberry Pi OS 64-bit before entering voice
mode. The expected import name is:

```python
import ten_vad_python
```

Recommended runtime settings:

```env
TEN_VAD_HOP_SIZE=256
TEN_VAD_START_THRESHOLD=0.50
TEN_VAD_END_THRESHOLD=0.35
VAD_START_MS=300
VAD_SILENCE_MS=900
```

Use the probe to tune thresholds in your room:

```bash
python vad_probe.py --duration 8
```

If speech starts before you talk, raise `TEN_VAD_START_THRESHOLD` or
`VAD_START_MS`. If speech is not detected, lower `TEN_VAD_START_THRESHOLD`.

## Raspberry Pi Audio

The expected microphone is a ReSpeaker 2-Mics Pi HAT. `arecord` can address it
as `plughw:3,0`, while `python-sounddevice` usually sees it by card name:

```env
RECORD_DEVICE=seeed2micvoicec
RECORD_SAMPLE_RATE=16000
RECORD_CHANNELS=1
```

Verify ALSA recording directly:

```bash
arecord -D plughw:3,0 -f S16_LE -r 16000 -c 1 -d 5 test.wav
aplay test.wav
```

Verify the Python audio path:

```bash
python pi_audio_check.py
```

## Run The Agent

```bash
python main.py
```

Commands:

- `voice` / `v`: enter continuous voice mode
- `exit` / `quit`: stop the program
- `reset` / `clear`: clear the current conversation

In voice mode, the program records one utterance, uploads it to the remote ASR
server, prints the recognized text, sends it to the LLM, and prints the reply.
Say `退出语音模式` to leave voice mode.

## Remote ASR Server

Run this on the GPU server, not on the Raspberry Pi:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-asr-server.txt
export ASR_MODEL_SIZE=medium
export ASR_DEVICE=cuda
export ASR_COMPUTE_TYPE=float16
export ASR_LANGUAGE=zh
python -m uvicorn asr_server:app --host 0.0.0.0 --port 8000
```

`medium + cuda + float16` is the recommended starting point for an RTX 4060.
Use `small` for lower latency, or `large-v3` if accuracy matters more than
response time.

Optional LAN token:

```bash
export ASR_SERVER_API_KEY=change-me
```

If the server token is set, put the same value in the Pi client's
`ASR_REMOTE_API_KEY`.

Test the server while bypassing proxy variables:

```bash
curl --noproxy '*' -F "file=@recordings/input.wav" -F "language=zh" http://192.168.1.100:8000/v1/transcriptions
```

If `/health` works but transcription returns `502`, check whether the Pi is
sending LAN traffic through a proxy:

```bash
env | grep -i proxy
curl -v --noproxy '*' http://192.168.1.100:8000/health
```

## Files

- `main.py`: command-line chat and voice loop
- `agent.py`: OpenAI-compatible chat API calls and error handling
- `conversation.py`: bounded conversation history
- `config.py`: environment variable loading and validation
- `recorder.py`: ReSpeaker microphone recording and WAV saving
- `voice_activity.py`: TEN VAD integration
- `asr.py`: remote ASR client plus optional faster-whisper provider
- `asr_server.py`: LAN HTTP ASR server for GPU-backed transcription
- `pi_audio_check.py`: Raspberry Pi microphone diagnostics
- `vad_probe.py`: TEN VAD probability measurement tool
- `.env.example`: local configuration template; never commit real `.env` values
