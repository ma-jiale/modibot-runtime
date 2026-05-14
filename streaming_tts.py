import asyncio
from collections.abc import AsyncIterator, Iterator

from tts import TextToSpeechProvider, play_audio_bytes_blocking


SENTENCE_ENDINGS = set("\u3002\uff01\uff1f!?;\uff1b\n")
SOFT_BREAKS = set("\uff0c,\u3001\uff1a: ")


class SentenceChunker:
    """Incrementally split streamed text into short speakable chunks."""

    def __init__(self, max_chars: int) -> None:
        """Create a chunker that flushes by punctuation or MAX_CHARS."""
        self._max_chars = max_chars
        self._buffer = ""

    def push(self, text: str) -> list[str]:
        """Add TEXT and return all newly completed chunks."""
        self._buffer += text
        chunks: list[str] = []

        while True:
            boundary = self._find_boundary()
            if boundary is None:
                return chunks

            chunk = self._buffer[:boundary].strip()
            self._buffer = self._buffer[boundary:].lstrip()
            if chunk:
                chunks.append(chunk)

    def flush(self) -> str | None:
        """Return the final incomplete chunk, if any."""
        chunk = self._buffer.strip()
        self._buffer = ""
        return chunk or None

    def _find_boundary(self) -> int | None:
        """Return the next chunk boundary index, or None."""
        for index, char in enumerate(self._buffer, start=1):
            if char in SENTENCE_ENDINGS:
                return index

        if len(self._buffer) < self._max_chars:
            return None

        return self._last_soft_break_before_limit() or self._max_chars

    def _last_soft_break_before_limit(self) -> int | None:
        """Return a soft punctuation boundary near the configured limit."""
        search_area = self._buffer[: self._max_chars]
        for index in range(len(search_area) - 1, -1, -1):
            if search_area[index] in SOFT_BREAKS:
                return index + 1
        return None


async def speak_streaming_response(
    text_stream: Iterator[str] | AsyncIterator[str],
    provider: TextToSpeechProvider,
    max_chunk_chars: int,
) -> str:
    """Stream text into TTS audio bytes and play them in order.

    Text, synthesis, and playback are separate stages connected by queues. The
    full visible reply is returned so callers can keep terminal behavior simple.
    """
    text_queue: asyncio.Queue[str | None] = asyncio.Queue()
    audio_queue: asyncio.Queue[bytes | None] = asyncio.Queue()
    player_task = asyncio.create_task(_audio_player(audio_queue))
    tts_task = asyncio.create_task(_tts_worker(text_queue, audio_queue, provider))

    full_reply = await _feed_text_chunks(text_stream, text_queue, max_chunk_chars)

    await tts_task
    await player_task
    return full_reply


async def _feed_text_chunks(
    text_stream: Iterator[str] | AsyncIterator[str],
    text_queue: asyncio.Queue[str | None],
    max_chunk_chars: int,
) -> str:
    """Print streamed tokens and queue speakable text chunks."""
    chunker = SentenceChunker(max_chunk_chars)
    full_reply: list[str] = []

    async for token in _as_async_iter(text_stream):
        print(token, end="", flush=True)
        full_reply.append(token)

        for sentence in chunker.push(token):
            await text_queue.put(sentence)

    leftover = chunker.flush()
    if leftover:
        await text_queue.put(leftover)
    await text_queue.put(None)
    return "".join(full_reply)


async def _tts_worker(
    text_queue: asyncio.Queue[str | None],
    audio_queue: asyncio.Queue[bytes | None],
    provider: TextToSpeechProvider,
) -> None:
    """Convert queued text chunks to WAV bytes."""
    while True:
        text = await text_queue.get()
        try:
            if text is None:
                await audio_queue.put(None)
                return

            audio = await asyncio.to_thread(provider.synthesize_to_bytes, text)
            await audio_queue.put(audio)
        finally:
            text_queue.task_done()


async def _audio_player(audio_queue: asyncio.Queue[bytes | None]) -> None:
    """Play queued WAV bytes in order."""
    while True:
        audio = await audio_queue.get()
        try:
            if audio is None:
                return
            await asyncio.to_thread(play_audio_bytes_blocking, audio)
        finally:
            audio_queue.task_done()


async def _as_async_iter(stream: Iterator[str] | AsyncIterator[str]) -> AsyncIterator[str]:
    """Adapt a sync or async iterator into an async iterator."""
    if hasattr(stream, "__aiter__"):
        async for item in stream:  # type: ignore[union-attr]
            yield item
        return

    for item in stream:  # type: ignore[union-attr]
        yield item
        await asyncio.sleep(0)
