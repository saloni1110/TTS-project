import os
import librosa
import numpy as np

def wav_to_mel_spectrogram(wav_path, sr=22050, n_mels=80, n_fft=1024, hop_length=256):
    y, sr = librosa.load(wav_path, sr=sr)
    mel_spectrogram = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels, power=1.0
    )
    mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)
    return mel_spectrogram_db

def process_audio_folder(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    for filename in os.listdir(input_dir):
        if filename.endswith(".wav"):
            wav_path = os.path.join(input_dir, filename)
            mel = wav_to_mel_spectrogram(wav_path)
            mel_filename = filename.replace(".wav", ".npy")
            np.save(os.path.join(output_dir, mel_filename), mel)
            print(f"Saved mel spectrogram: {mel_filename}")

if __name__ == "__main__":
     process_audio_folder("audio", "mels")
