use std::path::PathBuf;
use std::time::Instant;

use transcribe_rs::onnx::voxtral::{VoxtralModel, VoxtralParams};
use transcribe_rs::onnx::Quantization;

fn get_audio_duration(path: &PathBuf) -> Result<f64, Box<dyn std::error::Error>> {
    let reader = hound::WavReader::open(path)?;
    let spec = reader.spec();
    Ok(reader.duration() as f64 / spec.sample_rate as f64)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::init();

    let args: Vec<String> = std::env::args().collect();
    let model_path = args
        .get(1)
        .cloned()
        .unwrap_or_else(|| "models/voxtral-mini-3b".to_string());
    let wav_path = args
        .get(2)
        .cloned()
        .unwrap_or_else(|| "samples/jfk.wav".to_string());
    let language = args.get(3).cloned();

    let model_path = PathBuf::from(model_path);
    let wav_path = PathBuf::from(wav_path);

    let audio_duration = get_audio_duration(&wav_path)?;
    println!("Audio duration: {:.2}s", audio_duration);

    let load_start = Instant::now();
    let mut model = match VoxtralModel::load(&model_path, &Quantization::Int4) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("Int4 load failed: {e}; retrying FP16");
            match VoxtralModel::load(&model_path, &Quantization::FP16) {
                Ok(m) => m,
                Err(e) => {
                    eprintln!("FP16 load failed: {e}; retrying FP32");
                    VoxtralModel::load(&model_path, &Quantization::FP32)?
                }
            }
        }
    };
    println!("Model loaded in {:.2?}", load_start.elapsed());

    let samples = transcribe_rs::audio::read_wav_samples(&wav_path)?;
    let transcribe_start = Instant::now();
    let result = model.transcribe_with(
        &samples,
        &VoxtralParams {
            language,
            max_new_tokens: 256,
        },
    )?;
    let transcribe_duration = transcribe_start.elapsed();

    println!("Transcription completed in {:.2?}", transcribe_duration);
    println!(
        "Real-time speedup: {:.2}x",
        audio_duration / transcribe_duration.as_secs_f64()
    );
    println!("Transcription result:\n{}", result.text);

    Ok(())
}
