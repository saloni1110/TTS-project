# README.md 
# Text-to-Speech (TTS) Project

This project implements a basic Text-to-Speech (TTS) system consisting of data preprocessing, training an acoustic model, and inference with waveform synthesis.

---

## Project Structure

- `preprocess_audio.py`: Converts raw `.wav` audio files to mel spectrogram `.npy` files.
- `preprocess_text.py`: Normalizes input text and converts to sequence IDs; generates `processed_metadata.csv`.
- `tts.py`: Core module with text processing functions, acoustic model definition (encoder-decoder), and waveform synthesis (Griffin-Lim vocoder).
- `train_tts.py`: Loads processed metadata and mel data, prepares datasets, trains the acoustic TTS model, and saves checkpoints.
- `inference_tts.py`: Loads trained model weights to synthesize mel spectrograms and convert them to waveform audio from new text input.
- `checkpoints/`: Directory to save trained model weights.

---

## Getting Started

### Prerequisites

Install Python and required packages (check `requirements.txt`): pip install -r requirements.txt

## Usage

### Prepare Data
1. Place `.wav` files in `data/audio/`  
2. Run `preprocess_audio.py`  
3. Run `preprocess_text.py` to generate processed metadata  

### Train Model
python train_tts.py

### Inference
python inference_tts.py
