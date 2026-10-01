"""voice.stt: speech-to-text.

Phase 1 chose **Gemma audio** over whisper.cpp (0.29 s vs 0.92 s, no extra
model resident), so the runtime backend posts 16 kHz mono WAV to llama-server's
``/v1/chat/completions`` as ``input_audio``. ``FakeStt`` covers laptop tests.
"""

from __future__ import annotations

import base64
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


class Stt(Protocol):
    """A mono int16 clip -> transcript text."""

    def transcribe(self, pcm: np.ndarray, sample_rate: int = 16000) -> str: ...


class GemmaStt:
    """Transcribe through llama-server's Gemma audio projector."""

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
        # Gemma is a thinking model: the answer can land in either field.
        text = (message.get("content") or message.get("reasoning_content") or "").strip()
        return text.strip().strip('"').strip()

    def close(self) -> None:
        self._client.close()


class FakeStt:
    """Returns a fixed transcript (laptop tests / sim)."""

    def __init__(self, text: str = ""):
        self.text = text
        self.calls = 0

    def transcribe(self, pcm: np.ndarray, sample_rate: int = 16000) -> str:
        self.calls += 1
        return self.text
