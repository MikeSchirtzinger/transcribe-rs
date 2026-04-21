mod common;

use std::path::PathBuf;

use transcribe_rs::onnx::voxtral::{VoxtralModel, VoxtralParams};
use transcribe_rs::onnx::Quantization;

#[test]
fn test_voxtral_transcribe_jfk() {
    let _ = env_logger::try_init();

    let model_dir = PathBuf::from("models/voxtral-mini-3b");
    let wav_path = PathBuf::from("samples/jfk.wav");

    if !common::require_paths(&[&model_dir, &wav_path]) {
        return;
    }

    let mut model = VoxtralModel::load(&model_dir, &Quantization::Int4)
        .or_else(|_| VoxtralModel::load(&model_dir, &Quantization::FP16))
        .or_else(|_| VoxtralModel::load(&model_dir, &Quantization::FP32))
        .expect("Failed to load Voxtral model");

    let samples = transcribe_rs::audio::read_wav_samples(&wav_path).expect("read wav");

    let result = model
        .transcribe_with(
            &samples,
            &VoxtralParams {
                language: Some("en".into()),
                max_new_tokens: 256,
            },
        )
        .expect("transcribe");

    assert!(!result.text.is_empty(), "Transcription should not be empty");
    println!("Transcription: {}", result.text);
}
