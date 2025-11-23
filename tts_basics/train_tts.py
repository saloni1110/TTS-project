import os
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split

from tts import (
    char_to_id,
    build_simple_tts_model,
)


# --- Data Loader and Processing ---
mels_dir = "mels"
metadata_path = "processed_metadata.csv"

metadata = pd.read_csv(metadata_path)
metadata['filename'] = metadata['filename'].str.strip()  # Clean filenames
metadata['sequence'] = metadata['sequence'].apply(eval)  # Convert sequences from string to list


def load_mel(filename):
    mel_path = os.path.join(mels_dir, filename.replace('.wav', '.npy'))
    mel = np.load(mel_path)
    return mel.T.astype(np.float32)


def data_generator(dataframe):
    for _, row in dataframe.iterrows():
        text_seq = np.array(row['sequence'], dtype=np.int32)
        mel = load_mel(row['filename'])
        yield (text_seq, mel)


def tf_data_gen(dataframe):
    for text_seq, mel in data_generator(dataframe):
        decoder_input = mel[:-1]
        decoder_target = mel[1:]
        yield (text_seq, decoder_input), decoder_target


# Split data
train_df, val_df = train_test_split(metadata, test_size=0.1, random_state=42)

# Determine mel dimension from a sample file
sample_mel = load_mel(train_df.iloc[0]['filename'])
mel_dim = sample_mel.shape[1]

# Define output_signature with variable time dimension for text and mel
output_signature = (
    (
        tf.TensorSpec(shape=(None,), dtype=tf.int32),           # text sequences variable length
        tf.TensorSpec(shape=(None, mel_dim), dtype=tf.float32)  # decoder input mel variable length
    ),
    tf.TensorSpec(shape=(None, mel_dim), dtype=tf.float32)      # decoder target mel variable length
)

BATCH_SIZE = 16


def prepare_dataset(df):
    dataset = tf.data.Dataset.from_generator(
        lambda: tf_data_gen(df),
        output_signature=output_signature
    )
    dataset = dataset.padded_batch(
        BATCH_SIZE,
        padded_shapes=(
            (tf.TensorShape([None]), tf.TensorShape([None, mel_dim])),
            tf.TensorShape([None, mel_dim])
        ),
        padding_values=(
            (0, 0.0),
            0.0
        )
    ).prefetch(tf.data.AUTOTUNE)
    return dataset


train_dataset = prepare_dataset(train_df)
val_dataset = prepare_dataset(val_df)

# Build model using imported function from tts.py
vocab_size = max(char_to_id.values()) + 1
model = build_simple_tts_model(vocab_size, mel_dim=mel_dim)

# Checkpoint callback - must end with .weights.h5 when save_weights_only=True
checkpoint_path = "checkpoints/tts_model.weights.h5"
os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path,
    save_weights_only=True,
    save_best_only=True,
    monitor='val_loss',
    verbose=1
)

# Train model
EPOCHS = 50
model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=EPOCHS,
    callbacks=[checkpoint_callback]
)
