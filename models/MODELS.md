# MODELS.md

Filled by scripts/download_models.sh: filename, source URL, SHA256, and the pinned llama.cpp commit.

| File | Source | SHA256 |
|---|---|---|

## Downloaded 2026-09-27
| File | SHA256 |
|---|---|
| gemma/gemma-4-E2B-it-Q4_K_M.gguf | 740185b21d22ceb83a11c3aa62ad5842ef32c70f6096d756bbee85a1e4ec34b8 |
| gemma/mmproj-gemma4-e2b-f16.gguf | 140be8d7849741f88c50757d529b84373ee8e27052cc2236855b537f4a8215fa |
| gemma/mmproj-gemma4-e2b-q8_0.gguf | 9406f99c16d68cda4f1f0552192dcc99021ea1fc6d2fd50b1dc3ccf30d04b292 |
| whisper/ggml-base.en.bin | a03779c86df3323075f5e796cb2ce5029f00ec8869eee3fdfb897afe36c6d002 |
| piper/en_US-lessac-medium.onnx | 5efe09e69902187827af646e1a6e9d269dee769f9877d17b16b1b46eeaaf019f |
| yolo11n.pt | 0ebbc80d4a7680d14987a577cd21342b65ecfd94632bd9a8da63ae6417644ee1 |
| moondream/moondream2-text-model.gguf | e554c6b9de016673fd2c732e0342967727e9659ca5f853a4947cc96263fa602b |
| moondream/moondream2-mmproj.gguf | 4cc1cb3660d87ff56432ebeb7884ad35d67c48c7b9f6b2856f305e39c38eed8f |

llama.cpp build commit: d2e54583c7452353eb35d40431281f6ee984332f

## Evaluated 2026-09-29 (G1 fallback #4 — rejected)
| File | SHA256 | Note |
|---|---|---|
| gemma/gemma-4-E2B-it-Q3_K_M.gguf | 086e2f5ba85057f8f19712e3160a644728f74f323c9feeac4cd73fab11b43085 | frees ~520 MB but degrades vision (blue bus → "white and green"); Q4_K_M retained |

## Moonshine Voice STT — default backend (B3, 2026-10-02)
Package pinned: **`moonshine-voice==0.1.5`** (extra `moonshine`). Fetched by
`scripts/download_models.sh` from `https://download.moonshine.ai/model/...` into
`models/moonshine/`; only OnnxRuntime flatbuffers (`.ort`) are accepted.
Default model: **tiny-streaming-en**; **small-streaming-en** is the alternative
(real-mic results decide — see STATUS §Phase 4 voice v2). Paths below are
relative to `models/moonshine/download.moonshine.ai/model/`.

| File | SHA256 |
|---|---|
| tiny-streaming-en/quantized_26_08_21/encoder.ort | a8414e1a5dedf9f2093d7680601dd8a9b0433e7020260eafe0e370ead91134ca |
| tiny-streaming-en/quantized_26_08_21/adapter.ort | 22ecc949e146c49667fda28d102d4e30749a107dc88a396292aa8f277ef1347c |
| tiny-streaming-en/quantized_26_08_21/cross_kv.ort | 143a36667b8d05fd9d04e8c337b7ee121f37ef299aea6b3d82bdb3d3401950b4 |
| tiny-streaming-en/quantized_26_08_21/decoder_kv.ort | 8852553f312adb6c9aa4d17418015049b30f412209ee569d336548c0044627de |
| tiny-streaming-en/quantized_26_08_21/frontend.model.ort | 5121b561417b638afce0c6c31b760e37c93cf97f80d9b0031aad1fe7b6f25d61 |
| tiny-streaming-en/quantized_26_08_21/frontend.weights.ort | 217da24ac6f522ebf02da8ef288e77d1ac68d50d4a6821433182e4fbf4204bbd |
| small-streaming-en/quantized_26_08_21/encoder.ort | 2d4d973e91e8aca08c51e7e7efa28a46ab265b63d809d5294d18b86bcd85b993 |
| small-streaming-en/quantized_26_08_21/adapter.ort | c665f742364febad597cc9ac1e0b341ffbee0e24a1466e2f3bde95e6e4771762 |
| small-streaming-en/quantized_26_08_21/cross_kv.ort | e2d3417144e9514055ebfefe8dcc4c0a55a55adcb8530435844c75c53e352bf6 |
| small-streaming-en/quantized_26_08_21/decoder_kv.ort | 1a05465b1dd955858dfcbee039c0020fb5dd982b0f5094c34e61735d518d771b |
| small-streaming-en/quantized_26_08_21/frontend.model.ort | 09b1210ae30dc5f0f3e45f0ebab914c254741323114f53fbbe5ae62cca35058f |
| small-streaming-en/quantized_26_08_21/frontend.weights.ort | 7ef97521bd4bad3928f5bb6808586f4fcc6e92bd5990394112eed7d4052ec338 |
| spelling-en/spelling_cnn.ort | 79abb40b6849cf7c013e83a3f276a0d822bb94c3edcc9ad003d38815c2cf37d4 |

(medium-streaming-en is also downloaded for benchmarking; not pinned as a runtime
default.)
