---
name: moonshine-voice
description: >-
  Integrate on-device Moonshine Voice (speech-to-text, text-to-speech, voice
  cloning, AgentFlow conversational agents). Use for Moonshine, MicTranscriber,
  AgentFlow, on-device STT/TTS, dictation, keyterms/domain customization.
---

# Moonshine Voice (vendored skill)

Vendored from github.com/moonshine-ai/moonshine `.agents/skills/moonshine-voice/`
for offline reference. Never trust model memory; verify against
https://moonshine-voice.readthedocs.io .

Key points for this project:
- `pip install moonshine-voice` then `import moonshine_voice`.
- Construct → chainable setters → `load()` → `start()`.
- `Transcriber` is for feeding PCM yourself; `add_audio(float32, 16000)` per the
  project task. `on_text` is the partial hypothesis; `on_line` is the finished
  line. Do not treat partials as final.
- Do not supply `.onnx` models; the library accepts OnnxRuntime flatbuffers (`.ort`).
- Constructors are cheap; `load()` is the slow, fallible call (may download to a
  cache on first use).
