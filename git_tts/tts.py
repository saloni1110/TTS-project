import re
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense
from tensorflow.keras.models import Model
import librosa
import soundfile as sf

# --- Text Processing ---
characters = "abcdefghijklmnopqrstuvwxyz1234567890' "
char_to_id = {c: i + 1 for i, c in enumerate(characters)}  # 0 is reserved

def normalize_text(text):
    text = text.lower()
    text = text.strip()
    text = re.sub(r"[^a-z0-9' ]+", "", text)
    text = re.sub(r"\s+", " ", text)
    return text

def text_to_sequence(text):
    return [char_to_id.get(c, 0) for c in text]

# --- Acoustic Model (Encoder-Decoder) ---
def build_simple_tts_model(vocab_size, embedding_dim=256, lstm_units=256, mel_dim=80):
    # Encoder
    text_input = Input(shape=(None,), name="text_input")
    x = Embedding(vocab_size, embedding_dim)(text_input)
    encoder_output, state_h, state_c = LSTM(lstm_units, return_state=True)(x)
    encoder_states = [state_h, state_c]

    # Decoder (Simplified, teacher forcing assumed externally)
    decoder_input = Input(shape=(None, mel_dim), name="decoder_input")
    decoder_lstm = LSTM(lstm_units, return_sequences=True, return_state=True)
    decoder_output, _, _ = decoder_lstm(decoder_input, initial_state=encoder_states)
    mel_output = Dense(mel_dim)(decoder_output)

    model = Model([text_input, decoder_input], mel_output)
    model.compile(optimizer='adam', loss='mse')
    return model

# --- Waveform Synthesis (Griffin-Lim vocoder) ---
def mel_spectrogram_to_waveform(mel_spectrogram, sr=22050, n_fft=1024, hop_length=256):
    # librosa expects mel spectrogram in power scale, ensure proper conversion
    mel_spectrogram = np.maximum(mel_spectrogram, 1e-10)  # avoid log(0)
    mel_spectrogram = librosa.db_to_power(mel_spectrogram)
    waveform = librosa.feature.inverse.mel_to_audio(
        mel_spectrogram, sr=sr, n_fft=n_fft, hop_length=hop_length, n_iter=60)
    return waveform

# --- Example Run ---
if __name__ == "__main__":
    # Sample input text
    raw_text = "Hello, this is a test for simple TTS."

    # Text processing
    normalized = normalize_text(raw_text)
    sequence = text_to_sequence(normalized)
    sequence = np.array(sequence)[np.newaxis, :]  # Batch dimension

    # Dummy decoder input (usually previous mel frames, here zeros)
    mel_dim = 80
    decoder_input_seq = np.zeros((1, 50, mel_dim))  # e.g., 50 frames

    # Build model
    vocab_size = len(char_to_id) + 1
    model = build_simple_tts_model(vocab_size, mel_dim=mel_dim)

    # Dummy prediction (for demonstration; normally trained model output)
    predicted_mel = model.predict([sequence, decoder_input_seq])[0]

    # Convert mel spectrogram to waveform
    waveform = mel_spectrogram_to_waveform(predicted_mel.T)  # Transpose for librosa

    # Save to WAV
    sf.write("output.wav", waveform, 22050)
    print("Synthesized speech saved as output.wav")
