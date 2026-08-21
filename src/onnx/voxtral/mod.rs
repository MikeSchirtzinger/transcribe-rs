//! Voxtral speech-to-text (Mistral Voxtral-Mini-3B-2507 family).
//!
//! Voxtral is an audio-language model: a Whisper-style audio encoder whose
//! output is spliced directly into a Mistral decoder's `inputs_embeds`. The
//! ONNX export published at
//! [`onnx-community/Voxtral-Mini-3B-2507-ONNX`](https://huggingface.co/onnx-community/Voxtral-Mini-3B-2507-ONNX)
//! is organised as three graphs:
//!
//! - `audio_encoder[.quant].onnx` — mel features → audio embeddings (already
//!   projected to the decoder's 3072-dim hidden size).
//! - `embed_tokens[.quant].onnx` — token ids → text embeddings.
//! - `decoder_model_merged[.quant].onnx` — Mistral decoder that consumes
//!   pre-embedded `inputs_embeds` plus `past_key_values`.
//!
//! This module loads all three sessions, tokenises the standard transcription
//! prompt (`<s>[INST][AUDIO]lang:<L> [TRANSCRIBE][/INST]`), splices the audio
//! embeddings over the `[AUDIO]` placeholder tokens, and runs greedy decoding.

mod decoder;
mod mel;

use std::path::{Path, PathBuf};
use std::time::Instant;

use ndarray::{Array, Array2, Array3};
use ort::session::Session;
use ort::value::Value;
use tokenizers::Tokenizer;

use super::session as ort_session;
use super::Quantization;
use crate::{
    ModelCapabilities, SpeechModel, TranscribeError, TranscribeOptions, TranscriptionResult,
};

pub use decoder::{HIDDEN_SIZE, NUM_LAYERS};

/// Placeholder token id that gets replaced with audio encoder outputs. Matches
/// `audio_token_id` in `config.json`.
pub const AUDIO_TOKEN_ID: i64 = 24;

/// Default generation cap. 30 s of audio rarely exceeds a few hundred tokens,
/// but Voxtral's context window is 32k so we allow generous budget by default.
pub const DEFAULT_MAX_NEW_TOKENS: usize = 512;

const SUPPORTED_LANGUAGES: &[&str] = &["en", "fr", "de", "es", "it", "pt", "nl", "hi"];

const CAPABILITIES: ModelCapabilities = ModelCapabilities {
    name: "Voxtral Mini 3B",
    engine_id: "voxtral",
    sample_rate: 16000,
    languages: SUPPORTED_LANGUAGES,
    supports_timestamps: false,
    supports_translation: false,
    supports_streaming: false,
};

#[derive(Debug, Clone)]
pub struct VoxtralParams {
    /// BCP-47 language hint inserted into the prompt (e.g. `"en"`, `"fr"`).
    /// When `None`, Voxtral's automatic language detection is used by omitting
    /// the `lang:` prefix.
    pub language: Option<String>,
    /// Maximum number of decoder tokens to generate.
    pub max_new_tokens: usize,
}

impl Default for VoxtralParams {
    fn default() -> Self {
        Self {
            language: None,
            max_new_tokens: DEFAULT_MAX_NEW_TOKENS,
        }
    }
}

pub struct VoxtralModel {
    audio_encoder: Session,
    embed_tokens: Session,
    decoder: Session,
    tokenizer: Tokenizer,
    eos_token_id: i64,
}

impl VoxtralModel {
    /// Load a Voxtral model from a directory.
    ///
    /// Expected layout (matches the HuggingFace ONNX release):
    /// ```text
    /// model_dir/
    ///   onnx/
    ///     audio_encoder[.{quant}].onnx
    ///     audio_encoder[.{quant}].onnx_data   (ONNX external weights)
    ///     embed_tokens[.{quant}].onnx
    ///     decoder_model_merged[.{quant}].onnx
    ///   tokenizer.json
    ///   generation_config.json  (optional)
    /// ```
    ///
    /// The `quant` suffix follows transformers.js naming: `fp16`, `q4`,
    /// `q4f16`, `int8`, `quantized`, `bnb4`, `uint8`. Unknown variants fall
    /// back to the FP32 file.
    pub fn load(model_dir: &Path, quantization: &Quantization) -> Result<Self, TranscribeError> {
        Self::load_with_quants(model_dir, quantization, quantization)
    }

    /// Load with separate quantization for the audio encoder vs. the text
    /// stack (embed_tokens + decoder). Lets callers pair a CoreML-friendly
    /// FP16 encoder with a Q4 decoder (decoder FP16 exceeds ORT's 2 GB
    /// protobuf limit).
    pub fn load_with_quants(
        model_dir: &Path,
        encoder_quant: &Quantization,
        decoder_quant: &Quantization,
    ) -> Result<Self, TranscribeError> {
        if !model_dir.exists() {
            return Err(TranscribeError::ModelNotFound(model_dir.to_path_buf()));
        }

        let load_start = Instant::now();

        let onnx_dir = if model_dir.join("onnx").is_dir() {
            model_dir.join("onnx")
        } else {
            model_dir.to_path_buf()
        };

        let audio_encoder_path = resolve_onnx(&onnx_dir, "audio_encoder", encoder_quant);
        let embed_tokens_path = resolve_onnx(&onnx_dir, "embed_tokens", decoder_quant);
        let decoder_path = resolve_onnx(&onnx_dir, "decoder_model_merged", decoder_quant);

        for p in [&audio_encoder_path, &embed_tokens_path, &decoder_path] {
            if !p.exists() {
                return Err(TranscribeError::ModelNotFound(p.clone()));
            }
        }

        // Per-session EP strategy:
        // * audio_encoder — uses the global accelerator (e.g. CoreML on macOS).
        //   The encoder is a fixed 30 s graph, no KV state, and FP16 weights
        //   map cleanly to CoreML's Neural Engine / Metal.
        // * embed_tokens — lookup-heavy, trivially fast on CPU; no benefit
        //   from GPU dispatch.
        // * decoder_model_merged — stores weights in external `.onnx_data`
        //   sidecars, which CoreML's subgraph partitioner can't resolve
        //   (the subgraph path loses the base-path context at compile time,
        //   producing `initializer.cc !model_path.empty()` errors). Forcing
        //   CPU avoids that failure and side-steps the Q4 `MatMulNBits`
        //   custom op which CoreML has no native kernel for.
        log::info!("Loading Voxtral audio_encoder from {:?}", audio_encoder_path);
        let audio_encoder = ort_session::create_session(&audio_encoder_path)?;

        log::info!("Loading Voxtral embed_tokens from {:?}", embed_tokens_path);
        let embed_tokens = ort_session::create_session_cpu_only(&embed_tokens_path)?;

        log::info!("Loading Voxtral decoder from {:?}", decoder_path);
        let decoder = ort_session::create_session_cpu_only(&decoder_path)?;

        let tokenizer_path = model_dir.join("tokenizer.json");
        if !tokenizer_path.exists() {
            return Err(TranscribeError::ModelNotFound(tokenizer_path));
        }
        let tokenizer = Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| TranscribeError::Config(format!("tokenizer load: {e}")))?;

        // Resolve EOS from generation_config.json if present; default to 2.
        let eos_token_id = read_eos(&model_dir.join("generation_config.json")).unwrap_or(2);

        log::info!(
            "Voxtral model loaded in {:.2?} (eos={})",
            load_start.elapsed(),
            eos_token_id
        );

        Ok(Self {
            audio_encoder,
            embed_tokens,
            decoder,
            tokenizer,
            eos_token_id,
        })
    }

    pub fn transcribe_with(
        &mut self,
        samples: &[f32],
        params: &VoxtralParams,
    ) -> Result<TranscriptionResult, TranscribeError> {
        let total_start = Instant::now();

        // 1. Log-mel spectrogram: [N_MELS, N_FRAMES]
        let mel = mel::log_mel_spectrogram(samples);
        let mel_input: Array3<f32> = mel.insert_axis(ndarray::Axis(0));

        // 2. Audio encoder: [1, 128, 3000] -> [audio_seq, 3072]
        let audio_features = {
            let encode_start = Instant::now();
            let outputs = self
                .audio_encoder
                .run(ort::inputs!["audio_values" => Value::from_array(mel_input)?])?;
            let (shape, data) = outputs["audio_features"].try_extract_tensor::<f32>()?;
            // Expected shape is [audio_seq, 3072] (2D after the exporter flattens batch).
            if shape.len() != 2 {
                return Err(TranscribeError::Inference(format!(
                    "audio_features: expected rank 2, got {:?}",
                    shape
                )));
            }
            let audio_seq = shape[0] as usize;
            let hidden = shape[1] as usize;
            if hidden != HIDDEN_SIZE {
                return Err(TranscribeError::Inference(format!(
                    "audio_features hidden dim {} != {}",
                    hidden, HIDDEN_SIZE
                )));
            }
            log::debug!(
                "audio_encoder produced {} tokens in {:.2?}",
                audio_seq,
                encode_start.elapsed()
            );
            Array::from_shape_vec((audio_seq, hidden), data.to_vec())
                .map_err(|e| TranscribeError::Inference(format!("audio_features reshape: {e}")))?
        };

        // 3. Build prompt tokens surrounding N copies of AUDIO_TOKEN_ID.
        let audio_seq = audio_features.shape()[0];
        let input_ids = self.build_transcription_prompt(audio_seq, params.language.as_deref())?;

        log::debug!("Prompt length {} (audio tokens {})", input_ids.len(), audio_seq);

        // 4. Embed the prompt (including AUDIO placeholders — we overwrite
        //    those rows below).
        let text_embeds = {
            let ids_arr = Array2::from_shape_vec((1, input_ids.len()), input_ids.clone())?;
            let outputs = self
                .embed_tokens
                .run(ort::inputs!["input_ids" => Value::from_array(ids_arr)?])?;
            let (shape, data) = outputs["inputs_embeds"].try_extract_tensor::<f32>()?;
            let s3 = [shape[0] as usize, shape[1] as usize, shape[2] as usize];
            Array::from_shape_vec(s3, data.to_vec())
                .map_err(|e| TranscribeError::Inference(format!("text_embeds reshape: {e}")))?
        };

        // 5. Splice audio features over AUDIO placeholder positions.
        let combined =
            decoder::splice_audio_into_embeds(text_embeds, &input_ids, &audio_features)?;

        // 6. Autoregressive greedy decode.
        let decode_start = Instant::now();
        let generated = decoder::decode_greedy(
            &mut self.decoder,
            &mut self.embed_tokens,
            combined,
            params.max_new_tokens,
            self.eos_token_id,
        )?;
        log::debug!(
            "Generated {} tokens in {:.2?}",
            generated.len(),
            decode_start.elapsed()
        );

        // 7. Detokenise (strip special tokens).
        let ids_u32: Vec<u32> = generated.iter().map(|&t| t as u32).collect();
        let text = self
            .tokenizer
            .decode(&ids_u32, true)
            .map_err(|e| TranscribeError::Inference(format!("detokenise: {e}")))?;

        log::info!(
            "Voxtral transcribed in {:.2?}: \"{}\"",
            total_start.elapsed(),
            text
        );

        Ok(TranscriptionResult {
            text: text.trim().to_string(),
            segments: None,
        })
    }

    fn build_transcription_prompt(
        &self,
        audio_token_count: usize,
        language: Option<&str>,
    ) -> Result<Vec<i64>, TranscribeError> {
        // Rendered chat template (without [AUDIO], which is expanded manually):
        //   "<s>[INST]" + AUDIO × N + "lang:<L> [TRANSCRIBE][/INST]"
        //
        // For automatic language detection we drop the `lang:` prefix.
        let suffix_text = match language {
            Some(lang) => format!("lang:{} [TRANSCRIBE][/INST]", lang),
            None => "[TRANSCRIBE][/INST]".to_string(),
        };

        let prefix = self
            .tokenizer
            .encode("<s>[INST]", false)
            .map_err(|e| TranscribeError::Inference(format!("tokenize prefix: {e}")))?;
        let suffix = self
            .tokenizer
            .encode(suffix_text, false)
            .map_err(|e| TranscribeError::Inference(format!("tokenize suffix: {e}")))?;

        let prefix_ids = prefix.get_ids();
        let suffix_ids = suffix.get_ids();

        let mut ids: Vec<i64> = Vec::with_capacity(prefix_ids.len() + audio_token_count + suffix_ids.len());
        ids.extend(prefix_ids.iter().map(|&id| id as i64));
        ids.extend(std::iter::repeat(AUDIO_TOKEN_ID).take(audio_token_count));
        ids.extend(suffix_ids.iter().map(|&id| id as i64));
        Ok(ids)
    }
}

impl SpeechModel for VoxtralModel {
    fn capabilities(&self) -> ModelCapabilities {
        CAPABILITIES
    }

    fn transcribe_raw(
        &mut self,
        samples: &[f32],
        options: &TranscribeOptions,
    ) -> Result<TranscriptionResult, TranscribeError> {
        let params = VoxtralParams {
            language: options.language.clone(),
            ..Default::default()
        };
        self.transcribe_with(samples, &params)
    }
}

/// Resolve `{name}[.{suffix}].onnx` in the Voxtral ONNX directory, falling
/// back to FP32 if the requested variant is absent. Unlike the default
/// `Quantization` enum, transformers.js publishes more granular suffixes;
/// callers can still point at the FP32 file using `Quantization::FP32`.
fn resolve_onnx(dir: &Path, name: &str, quantization: &Quantization) -> PathBuf {
    let suffix = match quantization {
        Quantization::FP32 => None,
        Quantization::FP16 => Some("fp16"),
        Quantization::Int8 => Some("int8"),
        Quantization::Int4 => Some("q4"),
    };

    if let Some(s) = suffix {
        let candidate = dir.join(format!("{}_{}.onnx", name, s));
        if candidate.exists() {
            log::info!("Loading {} ({}): {}", name, s, candidate.display());
            return candidate;
        }
        log::warn!(
            "{} variant {} not found at {}, falling back to FP32",
            name,
            s,
            candidate.display()
        );
    }

    dir.join(format!("{}.onnx", name))
}

fn read_eos(path: &Path) -> Option<i64> {
    let raw = std::fs::read_to_string(path).ok()?;
    let v: serde_json::Value = serde_json::from_str(&raw).ok()?;
    v.get("eos_token_id").and_then(|x| x.as_i64())
}
