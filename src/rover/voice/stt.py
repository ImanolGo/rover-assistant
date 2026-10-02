"""voice.stt: speech-to-text backends behind ``voice.stt_backend``.

Default is **Moonshine streaming** (CPU, ONNX Runtime, no GPU); Gemma audio via
llama-server stays selectable.

Endpointer: the Silero VAD in :mod:`rover.voice.vad` owns turn end. Moonshine is
fed the endpointed turn and returns its final line; we do **not** also let
Moonshine's own line-completion end the turn (two endpointers would disagree).

``set_keyterms`` biases Moonshine toward the rover command vocabulary (intent
verbs, stop words, place/object names).
"""

from __future__ import annotations

import base64
import glob
import io
import wave
from typing import Protocol

import httpx
import numpy as np

TRANSCRIBE_PROMPT = "Transcribe this audio exactly. Output only the transcript."


def pcm16_to_wav_bytes(pcm: np.ndarray, sample_rate: int = 16000) -> bytes:
    """Wrap int16 mono PCM in a minimal WAV container (Gemma needs a real wav)."""
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(int(sample_rate))
        handle.writeframes(np.asarray(pcm, dtype=np.int16).tobytes())
    return buffer.getvalue()


def pcm16_to_base64(pcm: np.ndarray, sample_rate: int = 16000) -> str:
    return base64.b64encode(pcm16_to_wav_bytes(pcm, sample_rate)).decode("ascii")


def to_float32(pcm: np.ndarray) -> np.ndarray:
    """int16 PCM -> float32 in [-1, 1] (Moonshine's ``add_audio`` input)."""
    data = np.asarray(pcm)
    if data.dtype == np.int16:
        return data.astype(np.float32) / 32768.0
    return data.astype(np.float32)


def moonshine_dir(model: str) -> str:
    """Locate the local streaming-en model dir downloaded by ``download_models.sh``."""
    hits = sorted(
        glob.glob(f"models/moonshine/**/model/{model}-streaming-en/quantized_*", recursive=True)
    )
    if not hits:
        raise FileNotFoundError(f"moonshine model {model!r} not found under models/moonshine/")
    return hits[-1]


class Stt(Protocol):
    """A mono int16 clip -> transcript text."""

    def transcribe(self, pcm: np.ndarray, sample_rate: int = 16000) -> str: ...

    def close(self) -> None: ...


class GemmaStt:
    """Transcribe through llama-server's Gemma audio projector."""

    supports_streaming = False

    def __init__(self, url: str, timeout_s: float = 30.0, client: httpx.Client | None = None):
        self.url = url.rstrip("/")
        self._client = client or httpx.Client(timeout=timeout_s)

    def transcribe(self, pcm: np.ndarray, sample_rate: int = 16000) -> str:
        audio_url = f"data:audio/wav;base64,{pcm16_to_base64(pcm, sample_rate)}"
        payload = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_audio",
                            "input_audio": {"data": audio_url, "format": "wav"},
                        },
                        {"type": "text", "text": TRANSCRIBE_PROMPT},
                    ],
                }
            ],
            "max_tokens": 100,
            "temperature": 0.0,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        response = self._client.post(f"{self.url}/v1/chat/completions", json=payload)
        response.raise_for_status()
        message = response.json()["choices"][0]["message"]
        # Gemma is a thinking model; only the content is a transcript. The old
        # reasoning_content fallback is deliberately gone: thinking is not speech.
        return (message.get("content") or "").strip().strip('"').strip()

    def close(self) -> None:
        self._client.close()


class _MoonshineStream:
    """One turn's worth of streaming Moonshine audio (feed -> partials -> final)."""

    def __init__(self, stream: object):
        self._stream = stream
        self._partials: list[str] = []
        self._finals: list[str] = []

        def listener(event):
            # Duck-typed by class name so this module imports without moonshine.
            kind = type(event).__name__
            if kind == "LineCompleted":
                self._finals.append(event.line.text)
            elif kind == "LineUpdated":
                self._partials.append(event.line.text)

        stream.add_listener(listener)
        stream.start()

    def push(self, pcm: np.ndarray, sample_rate: int = 16000) -> str:
        self._stream.add_audio(to_float32(pcm).tolist(), sample_rate)
        return self.partial or ""

    def partial(self) -> str:
        return self._partials[-1].strip() if self._partials else ""

    def final(self) -> str:
        self._stream.stop()
        close = getattr(self._stream, "close", None)
        if close is not None:
            close()
        return " ".join(self._finals).strip() or self.partial()

    def cancel(self) -> None:
        stop = getattr(self._stream, "stop", None)
        if stop is not None:
            stop()
        close = getattr(self._stream, "close", None)
        if close is not None:
            close()


class MoonshineStt:
    """Streaming Moonshine (CPU). The model is loaded once; one stream per turn."""

    supports_streaming = True

    def __init__(
        self,
        model_dir: str,
        arch: str = "TINY_STREAMING",
        keyterms: list[str] | None = None,
        transcriber: object | None = None,
    ):
        if transcriber is None:
            from moonshine_voice import ModelArch
            from moonshine_voice.transcriber import Transcriber

            transcriber = Transcriber(str(model_dir), getattr(ModelArch, arch))
        self._transcriber = transcriber
        if keyterms:
            setter = getattr(self._transcriber, "set_keyterms", None)
            if setter is not None:
                setter(list(keyterms))

    def new_stream(self) -> _MoonshineStream:
        return _MoonshineStream(self._transcriber.create_stream())

    def transcribe(self, pcm: np.ndarray, sample_rate: int = 16000) -> str:
        stream = self.new_stream()
        data = to_float32(pcm)
        for start in range(0, len(data), 1280):
            stream.push(data[start : start + 1280], sample_rate)
        return stream.final()

    def close(self) -> None: ...


class FakeStt:
    """Returns a fixed transcript (laptop tests / sim)."""

    supports_streaming = False

    def __init__(self, text: str = ""):
        self.text = text
        self.calls = 0

    def transcribe(self, pcm: np.ndarray, sample_rate: int = 16000) -> str:
        self.calls += 1
        return self.text

    def close(self) -> None: ...


def moonshine_arch(model: str) -> str:
    """Model name -> Moonshine ``ModelArch`` member name."""
    return "SMALL_STREAMING" if model == "small" else "TINY_STREAMING"


def make_stt(config: object, url: str, keyterms: list[str] | None = None) -> Stt:
    """Build the configured backend: ``moonshine`` (default) or ``gemma``."""
    backend = getattr(config, "stt_backend", "moonshine")
    if backend == "moonshine":
        model = getattr(config, "moonshine_model", "tiny")
        return MoonshineStt(moonshine_dir(model), arch=moonshine_arch(model), keyterms=keyterms)
    return GemmaStt(url)
