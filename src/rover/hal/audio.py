"""hal.audio: microphone capture and speaker playback HAL.

Ports the legacy ROS2 audio nodes without ROS. The mic is a USB device
(``USB PnP Sound Device``) captured at 16 kHz mono through ``arecord`` in
80 ms frames; the speaker (``UACDemoV1.0``) is fed 48 kHz stereo S16 through
``aplay`` (PortAudio cannot open the card while pipewire holds it). Device
numbers move across boots, so both ends are selected by name substring.
Laptop/sim uses :class:`WavSource` and :class:`NullSpeaker`.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import threading
import time
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np

# card 1: Device [USB PnP Sound Device], device 0: USB Audio [USB Audio]
_ALSA_LINE = re.compile(r"card (\d+): ([^\[]*)\[([^\]]+)\], device (\d+): ([^\[]*)\[([^\]]+)\]")


@dataclass
class AudioConfig:
    """Tunables for the audio HAL (defaults mirror config/robot.yaml)."""

    mic_name: str = "USB PnP Sound Device"
    speaker_name: str = "UACDemoV1.0"
    mic_rate: int = 16000
    speaker_rate: int = 48000
    speaker_channels: int = 2
    frame_ms: int = 80

    @property
    def frame_samples(self) -> int:
        return int(self.mic_rate * self.frame_ms / 1000)


def parse_alsa_devices(text: str) -> list[dict[str, Any]]:
    """Parse ``arecord -l`` / ``aplay -l`` output into device records."""
    devices: list[dict[str, Any]] = []
    for line in text.splitlines():
        match = _ALSA_LINE.search(line)
        if match:
            card, _, device_name, device, _, device_sub = match.groups()
            devices.append(
                {
                    "card": int(card),
                    "device": int(device),
                    "name": device_name.strip(),
                    "subdevice": device_sub.strip(),
                }
            )
    return devices


def find_device(text: str, name: str) -> str | None:
    """Return ``plughw:<card>,<device>`` for the first name match, else None."""
    needle = name.lower()
    for device in parse_alsa_devices(text):
        haystack = f"{device['name']} {device['subdevice']}".lower()
        if needle in haystack:
            return f"plughw:{device['card']},{device['device']}"
    return None


def list_alsa_devices(flag: str = "-l", command: str = "arecord") -> str:
    """Run ``arecord -l`` / ``aplay -l`` and return the listing text."""
    if shutil.which(command) is None:
        return ""
    try:
        return subprocess.run([command, flag], capture_output=True, text=True, check=False).stdout
    except OSError:
        return ""


def normalize_int16(
    frame: np.ndarray, target_peak: int = 20000, max_gain: float = 15.0
) -> np.ndarray:
    """Boost a quiet int16 frame toward ``target_peak`` (legacy formula)."""
    peak = int(np.max(np.abs(frame))) if frame.size else 0
    if peak > 100:
        gain = min(target_peak / peak, max_gain)
        if gain > 1.0:
            return (frame * gain).astype(np.int16)
    return frame


def _resample_mono(samples: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    if source_rate == target_rate or samples.size == 0:
        return samples.astype(np.float32)
    count = int(len(samples) * target_rate / source_rate)
    resampled = np.interp(np.linspace(0, len(samples) - 1, count), np.arange(len(samples)), samples)
    return resampled.astype(np.float32)


def _to_float32(samples: np.ndarray) -> np.ndarray:
    if samples.dtype == np.int16:
        return samples.astype(np.float32) / 32768.0
    return samples.astype(np.float32)


def to_speaker_format(
    samples: np.ndarray, source_rate: int, config: AudioConfig, max_amplitude: float = 0.95
) -> np.ndarray:
    """Resample to the speaker rate, fan mono out to stereo, and peak-limit."""
    data = _to_float32(samples)
    if data.ndim == 1:
        data = _resample_mono(data, source_rate, config.speaker_rate)
        if config.speaker_channels >= 2:
            data = np.column_stack([data] * config.speaker_channels)
    else:
        if source_rate != config.speaker_rate:
            channels = [
                _resample_mono(data[:, ch], source_rate, config.speaker_rate)
                for ch in range(data.shape[1])
            ]
            data = np.column_stack(channels)
        if config.speaker_channels == 1:
            data = data.mean(axis=1)
    if data.size:
        peak = float(np.abs(data).max())
        if peak > max_amplitude:
            data = data * (max_amplitude / peak)
    return data.astype(np.float32)


class AudioCapture(Protocol):
    """Yields fixed-size mono int16 frames of ``config.frame_samples``."""

    def read_frame(self) -> np.ndarray | None: ...

    def close(self) -> None: ...


class Speaker(Protocol):
    """Plays float32 samples at ``rate``."""

    def play(self, samples: np.ndarray, rate: int) -> None: ...

    def close(self) -> None: ...


class WavSource:
    """Fake capture source backed by a WAV file (laptop / sim / tests)."""

    def __init__(self, path: str | Path, frame_samples: int = 1280, loop: bool = True):
        self.path = Path(path)
        self.frame_samples = frame_samples
        self.loop = loop
        self._wave = wave.open(str(self.path), "rb")
        if self._wave.getsampwidth() != 2:
            raise ValueError("WavSource requires 16-bit PCM WAV files")
        self.rate = self._wave.getframerate()
        self.channels = self._wave.getnchannels()

    def read_frame(self) -> np.ndarray | None:
        raw = self._wave.readframes(self.frame_samples)
        if len(raw) < self.frame_samples * self.channels * 2:
            if not self.loop:
                return None
            self._wave.rewind()
            raw = self._wave.readframes(self.frame_samples)
            if not raw:
                return None
        data = np.frombuffer(raw, dtype=np.int16)
        if self.channels > 1:
            data = data.reshape(-1, self.channels)[:, 0]
        return data.copy()

    def close(self) -> None:
        self._wave.close()


class AlsaCapture:
    """Real capture via ``arecord`` (pulse source when a sound server owns the card)."""

    def __init__(self, config: AudioConfig, device: str | None = None):
        self.config = config
        self.source = None if device else find_pulse_source(config.mic_name)
        self.device = device or (find_device(list_alsa_devices("-l"), config.mic_name) or "default")
        target, env = self.device, None
        if self.source:
            target = "pulse"
            env = {**os.environ, "PULSE_SOURCE": self.source}
        self._proc = subprocess.Popen(
            [
                "arecord",
                "-D",
                target,
                "-r",
                str(config.mic_rate),
                "-c",
                "1",
                "-f",
                "S16_LE",
                "-t",
                "raw",
                "--buffer-size=8192",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            env=env,
        )

    def read_frame(self) -> np.ndarray | None:
        needed = self.config.frame_samples * 2
        buffer = b""
        while len(buffer) < needed:
            chunk = self._proc.stdout.read(needed - len(buffer))
            if not chunk:
                return None
            buffer += chunk
        return np.frombuffer(buffer, dtype=np.int16).copy()

    def close(self) -> None:
        self._proc.terminate()
        try:
            self._proc.wait(timeout=1.0)
        except subprocess.TimeoutExpired:
            self._proc.kill()


class NullSpeaker:
    """Fake speaker that records what it was asked to play."""

    def __init__(self) -> None:
        self.played: list[np.ndarray] = []

    def play(self, samples: np.ndarray, rate: int) -> None:
        self.played.append(np.asarray(samples))

    def close(self) -> None: ...


def parse_pulse_sinks(text: str) -> list[dict[str, str]]:
    """Parse ``pactl list sinks short`` (tab-separated: index, name, driver, ...)."""
    sinks: list[dict[str, str]] = []
    for line in text.splitlines():
        parts = line.split("\t")
        if len(parts) >= 2:
            sinks.append({"index": parts[0].strip(), "name": parts[1].strip()})
    return sinks


def _normalise(text: str) -> str:
    """Lower-case alphanumerics only, so 'USB PnP Sound Device' matches underscores."""
    return re.sub(r"[^a-z0-9]", "", text.lower())


def _pactl_names(kind: str) -> list[str]:
    if shutil.which("pactl") is None:
        return []
    try:
        listing = subprocess.run(
            ["pactl", "list", kind, "short"], capture_output=True, text=True, check=False
        ).stdout
    except OSError:
        return []
    return [entry["name"] for entry in parse_pulse_sinks(listing)]


def find_pulse_sink(name: str) -> str | None:
    """Return the PulseAudio/PipeWire sink matching ``name`` (normalised)."""
    needle = _normalise(name)
    return next((n for n in _pactl_names("sinks") if needle in _normalise(n)), None)


def find_pulse_source(name: str) -> str | None:
    """Return the PulseAudio/PipeWire source matching ``name`` (normalised)."""
    needle = _normalise(name)
    return next((n for n in _pactl_names("sources") if needle in _normalise(n)), None)


class AlsaSpeaker:
    """Real playback through ``aplay``.

    Here pipewire/pulseaudio owns the UACDemo card, so direct ``plughw:0,0`` is
    busy and PortAudio cannot open it. We resolve the matching PulseAudio sink by
    name and stream raw 48 kHz stereo S16 to ``aplay -D pulse`` with
    ``PULSE_SINK`` set; with no sound server (headless) we fall back to the ALSA
    ``plughw`` device. ``aplay`` restarts once if the open races.
    """

    def __init__(self, config: AudioConfig, device: str | None = None):
        self.config = config
        self.sink = None if device else find_pulse_sink(config.speaker_name)
        self.device = device or (
            find_device(list_alsa_devices("-l", "aplay"), config.speaker_name) or "default"
        )
        self._lock = threading.Lock()
        self._proc: subprocess.Popen | None = None
        self._spawn()

    def _spawn(self) -> None:
        target, env = self.device, None
        if self.sink:
            target = "pulse"
            env = {**os.environ, "PULSE_SINK": self.sink}
        self._proc = subprocess.Popen(
            [
                "aplay",
                "-q",
                "-D",
                target,
                "-r",
                str(self.config.speaker_rate),
                "-c",
                str(self.config.speaker_channels),
                "-f",
                "S16_LE",
                "-t",
                "raw",
                "-",
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            env=env,
        )

    def play(self, samples: np.ndarray, rate: int) -> None:
        fmt = to_speaker_format(samples, rate, self.config)
        pcm = (np.clip(fmt, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()
        with self._lock:
            for attempt in range(2):
                if self._proc is None or self._proc.poll() is not None:
                    self._spawn()
                try:
                    self._proc.stdin.write(pcm)
                    self._proc.stdin.flush()
                    return
                except (BrokenPipeError, OSError):
                    if attempt == 1:
                        raise
                    time.sleep(0.2)

    def close(self) -> None:
        if self._proc is None:
            return
        try:
            self._proc.stdin.close()
        except Exception:  # noqa: BLE001
            pass
        try:
            self._proc.wait(timeout=2.0)
        except subprocess.TimeoutExpired:
            self._proc.kill()


def open_audio(config: AudioConfig | None = None, *, dry_run: bool = False):
    """Build ``(capture, speaker)`` for the configured backend."""
    config = config or AudioConfig()
    if dry_run:
        return None, NullSpeaker()
    return AlsaCapture(config), AlsaSpeaker(config)
