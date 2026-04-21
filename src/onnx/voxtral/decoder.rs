//! Autoregressive greedy decoder for Voxtral's `decoder_model_merged.onnx`.
//!
//! The decoder is a Mistral/Llama-style LM with 30 layers and grouped-query
//! attention (num_key_value_heads = 8, head_dim = 128). The ONNX merged model
//! accepts pre-embedded `inputs_embeds` together with the standard transformers
//! `past_key_values.{layer}.{key,value}` cache.

use std::borrow::Cow;

use ndarray::{Array, Array2, Array3, Array4};
use ort::session::Session;
use ort::value::{DynValue, Value};

use crate::TranscribeError;

use super::AUDIO_TOKEN_ID;

pub const NUM_LAYERS: usize = 30;
pub const NUM_KV_HEADS: usize = 8;
pub const HEAD_DIM: usize = 128;
pub const HIDDEN_SIZE: usize = 3072;

/// Build combined `inputs_embeds` by splicing encoder audio features into the
/// token embeddings at every position that holds [AUDIO] (`AUDIO_TOKEN_ID`).
pub fn splice_audio_into_embeds(
    mut text_embeds: Array3<f32>,
    input_ids: &[i64],
    audio_features: &Array2<f32>,
) -> Result<Array3<f32>, TranscribeError> {
    let seq_len = text_embeds.shape()[1];
    if input_ids.len() != seq_len {
        return Err(TranscribeError::Inference(format!(
            "input_ids length {} does not match text_embeds seq_len {}",
            input_ids.len(),
            seq_len
        )));
    }

    let audio_len = audio_features.shape()[0];
    let hidden = audio_features.shape()[1];
    if hidden != HIDDEN_SIZE {
        return Err(TranscribeError::Inference(format!(
            "audio feature dim {} != expected hidden {}",
            hidden, HIDDEN_SIZE
        )));
    }

    let mut audio_cursor = 0usize;
    for pos in 0..seq_len {
        if input_ids[pos] == AUDIO_TOKEN_ID {
            if audio_cursor >= audio_len {
                return Err(TranscribeError::Inference(format!(
                    "more AUDIO placeholders in input_ids than audio features ({} provided)",
                    audio_len
                )));
            }
            for h in 0..HIDDEN_SIZE {
                text_embeds[[0, pos, h]] = audio_features[[audio_cursor, h]];
            }
            audio_cursor += 1;
        }
    }

    if audio_cursor != audio_len {
        return Err(TranscribeError::Inference(format!(
            "audio features unused: placed {} of {}",
            audio_cursor, audio_len
        )));
    }

    Ok(text_embeds)
}

/// Run greedy autoregressive decoding until EOS or `max_new_tokens`. Returns
/// the generated token ids (excluding the prompt).
pub fn decode_greedy(
    decoder: &mut Session,
    embed_tokens: &mut Session,
    initial_embeds: Array3<f32>,
    max_new_tokens: usize,
    eos_token_id: i64,
) -> Result<Vec<i64>, TranscribeError> {
    let prompt_len = initial_embeds.shape()[1];
    let mut total_len = prompt_len;

    let mut inputs_embeds = initial_embeds;
    let mut attention_mask = Array2::<i64>::ones((1, total_len));
    let mut position_ids: Array2<i64> =
        Array2::from_shape_fn((1, prompt_len), |(_, j)| j as i64);

    let mut kv: Vec<Array4<f32>> = (0..NUM_LAYERS * 2)
        .map(|_| Array4::<f32>::zeros((1, NUM_KV_HEADS, 0, HEAD_DIM)))
        .collect();

    let mut generated: Vec<i64> = Vec::with_capacity(max_new_tokens);

    for step in 0..max_new_tokens {
        let mut ort_inputs: Vec<(Cow<'_, str>, DynValue)> =
            Vec::with_capacity(3 + NUM_LAYERS * 2);

        ort_inputs.push((
            "inputs_embeds".into(),
            Value::from_array(inputs_embeds.clone())?.into_dyn(),
        ));
        ort_inputs.push((
            "attention_mask".into(),
            Value::from_array(attention_mask.clone())?.into_dyn(),
        ));
        ort_inputs.push((
            "position_ids".into(),
            Value::from_array(position_ids.clone())?.into_dyn(),
        ));

        for layer in 0..NUM_LAYERS {
            let k_name = format!("past_key_values.{}.key", layer);
            let v_name = format!("past_key_values.{}.value", layer);
            ort_inputs.push((
                k_name.into(),
                Value::from_array(kv[layer * 2].clone())?.into_dyn(),
            ));
            ort_inputs.push((
                v_name.into(),
                Value::from_array(kv[layer * 2 + 1].clone())?.into_dyn(),
            ));
        }

        let outputs = decoder.run(ort_inputs)?;

        let next_token = {
            let (shape, data) = outputs["logits"].try_extract_tensor::<f32>()?;
            let seq = shape[1] as usize;
            let vocab = shape[2] as usize;
            let start = (seq - 1) * vocab;
            let slice = &data[start..start + vocab];
            argmax(slice) as i64
        };

        if next_token == eos_token_id {
            log::debug!("Hit EOS at step {}", step);
            break;
        }
        generated.push(next_token);

        // Refresh KV cache from present.*.
        for layer in 0..NUM_LAYERS {
            let k_name = format!("present.{}.key", layer);
            let v_name = format!("present.{}.value", layer);
            let (kshape, kdata) = outputs[k_name.as_str()].try_extract_tensor::<f32>()?;
            let (vshape, vdata) = outputs[v_name.as_str()].try_extract_tensor::<f32>()?;
            let kshape4 = [
                kshape[0] as usize,
                kshape[1] as usize,
                kshape[2] as usize,
                kshape[3] as usize,
            ];
            let vshape4 = [
                vshape[0] as usize,
                vshape[1] as usize,
                vshape[2] as usize,
                vshape[3] as usize,
            ];
            kv[layer * 2] = Array::from_shape_vec(kshape4, kdata.to_vec())
                .map_err(|e| TranscribeError::Inference(format!("KV key reshape: {e}")))?;
            kv[layer * 2 + 1] = Array::from_shape_vec(vshape4, vdata.to_vec())
                .map_err(|e| TranscribeError::Inference(format!("KV value reshape: {e}")))?;
        }

        // Prepare next-step inputs.
        total_len += 1;
        inputs_embeds = embed_single_token(embed_tokens, next_token)?;
        attention_mask = Array2::<i64>::ones((1, total_len));
        position_ids = Array2::from_shape_vec((1, 1), vec![(total_len - 1) as i64])?;
    }

    Ok(generated)
}

fn argmax(slice: &[f32]) -> usize {
    let mut best_i = 0usize;
    let mut best_v = f32::NEG_INFINITY;
    for (i, &v) in slice.iter().enumerate() {
        if v > best_v {
            best_v = v;
            best_i = i;
        }
    }
    best_i
}

fn embed_single_token(
    embed_tokens: &mut Session,
    token_id: i64,
) -> Result<Array3<f32>, TranscribeError> {
    let input_ids = Array2::from_shape_vec((1, 1), vec![token_id])?;
    let outputs = embed_tokens.run(ort::inputs!["input_ids" => Value::from_array(input_ids)?])?;
    let (shape, data) = outputs["inputs_embeds"].try_extract_tensor::<f32>()?;
    let s3 = [shape[0] as usize, shape[1] as usize, shape[2] as usize];
    Array::from_shape_vec(s3, data.to_vec())
        .map_err(|e| TranscribeError::Inference(format!("embed reshape: {e}")))
}
