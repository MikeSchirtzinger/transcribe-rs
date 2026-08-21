# Model weights

`models/` is gitignored and is not stored in this repository. Every file it held
came from HuggingFace and can be re-downloaded. This manifest records what was
there and which directory layout the loader expects.

Only `voxtral-mini-3b` is required by the Rust Voxtral ONNX engine. The GGUF
directories were used for llama.cpp comparison runs and are optional.

## voxtral-mini-3b (required, ~6.1G)

ONNX export of [mistralai/Voxtral-Mini-3B-2507](https://huggingface.co/mistralai/Voxtral-Mini-3B-2507),
community build published by `onnx-community`.

`VoxtralModel::load_with_quants` resolves three graphs out of `models/voxtral-mini-3b/onnx/`
by name plus quantization suffix, so keep these filenames:

```
onnx/audio_encoder_fp16.onnx          + .onnx_data, .onnx.external
onnx/audio_encoder_q4.onnx            + .onnx_data, .onnx.external
onnx/audio_encoder_q4_inline.onnx
onnx/decoder_model_merged_q4.onnx     + .onnx_data, .onnx_data_1
onnx/embed_tokens_q4.onnx             + .onnx_data
```

Root of the directory also needs `tokenizer.json`, `config.json`,
`generation_config.json`, `preprocessor_config.json`, `special_tokens_map.json`,
and `tokenizer_config.json`.

The default pairing is an FP16 encoder with a Q4 decoder. The decoder stays Q4
because an FP16 decoder exceeds the 2GB protobuf limit in ONNX Runtime, and it
loads CPU-only because CoreML's subgraph partitioner cannot resolve external
data sidecars.

## voxtral-gguf (optional, ~3.0G)

llama.cpp GGUF build of Voxtral Mini 3B.

```
Voxtral-Mini-3B-2507-Q4_K_M.gguf
mmproj-Voxtral-Mini-3B-2507-Q8_0.gguf
```

## voxtral-realtime-gguf (optional, ~30G)

GGUF conversions of [mistralai/Voxtral-Mini-4B-Realtime-2602](https://huggingface.co/mistralai/Voxtral-Mini-4B-Realtime-2602),
a different model family from the 3B the ONNX engine targets. Ten quantizations
of the same weights were downloaded for a size and quality sweep:

```
Q2_K  Q4_0  Q4_1  Q4_K  Q4_K_M  Q5_0  Q5_1  Q5_K  Q6_K  Q8_0
```

Re-download a single quantization rather than the full set unless you are
repeating the sweep.

## voxtral-realtime-acceldium (optional, ~5.3G)

A llama.cpp compatible implementation of Voxtral Realtime 4B, apache-2.0.

```
voxtral-realtime-4b-mmproj-f16.gguf
voxtral-realtime-4b-text-q8_0.gguf
```
