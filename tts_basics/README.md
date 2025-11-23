# README.md 
# Text-to-Speech (TTS) Project
This project contains a simple, educational Text-to-Speech (TTS) model built from scratch using PyTorch. This project implements a basic Text-to-Speech (TTS) system consisting of data preprocessing, training an acoustic model; converting text input to mel spectrograms and then reconstructing audio, and inference with waveform synthesis. It is designed for learning and experimentation, making it an excellent reference for beginners entering the TTS field.

Features: a. Minimal character-to-spectrogram neural architecture b. Data processing scripts for text and audio preparation c. Training pipeline with example dataset and evaluation outputs d. Step-by-step code documentation for easy understanding e. Ready-to-run demo and instructions

Goals: a. Help users understand the core concepts behind neural TTS b. Provide a hands-on, reproducible example for portfolio or academic use c. Serve as a starting point for more advanced neural voice synthesis projects

Contents: a. Python source code for model, training, and inference b. Sample dataset and usage instructions c. Detailed README with setup, usage, and results d. Guidance on further enhancements and deployment


---

## Project Structure

- `preprocess_audio.py`: Converts raw `.wav` audio files to mel spectrogram `.npy` files.
- `preprocess_text.py`: Normalizes input text and converts to sequence IDs; generates `processed_metadata.csv`.
- `tts.py`: Core module with text processing functions, acoustic model definition (encoder-decoder), and waveform synthesis (Griffin-Lim vocoder).
- `train_tts.py`: Loads processed metadata and mel data, prepares datasets, trains the acoustic TTS model, and saves checkpoints.
- `inference_tts.py`: Loads trained model weights to synthesize mel spectrograms and convert them to waveform audio from new text input.
- `checkpoints/`: Directory to save trained model weights.

---

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
