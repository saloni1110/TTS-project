import numpy as np
import tensorflow as tf
import librosa
import soundfile as sf
from tts import (
    normalize_text,
    text_to_sequence,
    char_to_id,
    build_simple_tts_model,
)
mel_dim = 80  # or dynamically set based on your mel files
# --- Step 1: Prepare input text sequence ---
def prepare_text_input(text):
    normalized = normalize_text(text)
    seq = text_to_sequence(normalized)
    return np.array(seq)[np.newaxis, :]  # Add batch dimension


# --- Step 2: Prepare decoder input seed --- 
def prepare_decoder_input(start_frames=1):
    return np.zeros((1, start_frames, mel_dim), dtype=np.float32)


# --- Step 3: Load trained weights and build model ---
vocab_size = max(char_to_id.values()) + 1  # Correctly get vocab size as int
model = build_simple_tts_model(vocab_size, mel_dim=mel_dim)
model.load_weights('checkpoints/tts_model.weights.h5')


# --- Step 4 & 5: Run inference to get mel spectrogram prediction ---
def infer_mel(text, max_len=200):
    text_input = prepare_text_input(text)
    decoder_input = prepare_decoder_input()
    mel_outputs = []

    for _ in range(max_len):
        pred = model.predict([text_input, decoder_input])
        next_frame = pred[:, -1:, :]
        mel_outputs.append(next_frame)
        decoder_input = np.concatenate([decoder_input, next_frame], axis=1)

    mel_spectrogram = np.concatenate(mel_outputs, axis=1)[0]  # Shape (time, mel_dim)
    return mel_spectrogram


# --- Step 6: Vocoder - synthesize waveform from mel spectrogram ---
def mel_to_audio_griffin_lim(mel_spectrogram, n_fft=1024, hop_length=256, n_iter=60):
    mel_spectrogram = mel_spectrogram.T  # Transpose to (n_mels, time)
    mel_basis = librosa.filters.mel(sr=22050, n_fft=n_fft, n_mels=mel_spectrogram.shape[0])
    inv_mel_basis = np.linalg.pinv(mel_basis)
    linear_spec = np.maximum(1e-10, np.dot(inv_mel_basis, mel_spectrogram))

    audio = librosa.griffinlim(linear_spec, n_iter=n_iter, hop_length=hop_length, win_length=n_fft)
    return audio


if __name__ == "__main__":
    input_text = "hello world, this is a test."
    mel_spectrogram = infer_mel(input_text)

    print(f"Predicted mel spectrogram shape: {mel_spectrogram.shape}")

    audio_waveform = mel_to_audio_griffin_lim(mel_spectrogram)

    sf.write("output.wav", audio_waveform, 22050)
    print("Audio saved to output.wav")
