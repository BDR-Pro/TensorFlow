import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow_datasets import load
import tensorflow_text as text  
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Dense, Embedding, MultiHeadAttention, Dropout, LayerNormalization, GlobalAveragePooling1D, Input
from tensorflow.keras.models import Model
from tensorflow.python.profiler.model_analyzer import profile
from tensorflow.python.profiler.option_builder import ProfileOptionBuilder

"""
This script builds and trains a Transformer model on the IMDb movie reviews dataset for sentiment analysis. 
It demonstrates the use of TensorFlow 2, TensorFlow Datasets (TFDS), and TensorFlow Text for efficient text tokenization.

The Transformer model architecture includes:
- Multi-head attention mechanisms,
- Positional embeddings, and
- Feed-forward networks.

TensorFlow Text is used here for tokenization, transforming raw text into a format that's suitable for the model.
"""

# Define the Transformer block
class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = tf.keras.Sequential([
            Dense(ff_dim, activation="relu"), 
            Dense(embed_dim),
        ])
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(rate)
        self.dropout2 = Dropout(rate)

    def call(self, inputs, training=False):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

# Define the token and position embedding layer
class TokenAndPositionEmbedding(tf.keras.layers.Layer):
    def __init__(self, maxlen, vocab_size, embed_dim):
        super(TokenAndPositionEmbedding, self).__init__()
        self.token_emb = Embedding(input_dim=vocab_size, output_dim=embed_dim)
        self.pos_emb = Embedding(input_dim=maxlen, output_dim=embed_dim)

    def call(self, x):
        positions = tf.range(start=0, limit=tf.shape(x)[-1], delta=1)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        return x + positions

# Model hyperparameters
vocab_size = 20000  # Top 20k words
maxlen = 200  # Consider the first 200 words of each review
embed_dim = 32  # Embedding size for each token
num_heads = 2  # Number of attention heads
ff_dim = 32  # Hidden layer size in feed-forward network inside transformer

# Build the model
inputs = Input(shape=(maxlen,))
embedding_layer = TokenAndPositionEmbedding(maxlen, vocab_size, embed_dim)
x = embedding_layer(inputs)
transformer_block = TransformerBlock(embed_dim, num_heads, ff_dim)
x = transformer_block(x)
x = GlobalAveragePooling1D()(x)
x = Dropout(0.1)(x)
x = Dense(20, activation="relu")(x)
x = Dropout(0.1)(x)
outputs = Dense(2, activation="softmax")(x)
model = Model(inputs=inputs, outputs=outputs)

# Compile the model
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

# Load the IMDB dataset
dataset, info = load("imdb_reviews", with_info=True, as_supervised=True)
train_dataset, test_dataset = dataset["train"], dataset["test"]

# Preprocessing and tokenization
tokenizer = text.WhitespaceTokenizer()

def build_vocabulary(data, vocab_size):
    """Build vocabulary from the dataset."""
    vocab = {}
    for text, _ in tfds.as_numpy(data.batch(1024)):
        for example_text in text:
            example_text = example_text.decode('utf-8').lower()  # Decode from bytes to string
            tokens = tokenizer.tokenize(example_text).numpy()  # Tokenize and convert to numpy
            for token in tokens:
                word = token.decode('utf-8')  # Decode tokens from bytes to string
                if word not in vocab:
                    vocab[word] = 1
                else:
                    vocab[word] += 1
    return {k: v for k, v in sorted(vocab.items(), key=lambda item: item[1], reverse=True)[:vocab_size]}

# Create a vocabulary
vocabulary = build_vocabulary(train_dataset, vocab_size)

def encode_map_fn(text, label):
    """Encode text data to integers."""
    text = tf.strings.lower(text)
    encoded_text = tokenizer.tokenize(text).to_tensor()  # Convert to tensor
    return encoded_text, label

batch_size = 32
# Prepare datasets for training and testing
train_data = train_dataset.map(encode_map_fn)
train_data = train_data.padded_batch(batch_size, padded_shapes=([None], []))
test_data = test_dataset.map(encode_map_fn)
test_data = test_data.padded_batch(batch_size, padded_shapes=([None], []))

# Train the model
history = model.fit(train_data, epochs=2, validation_data=test_data)

# Plot training and validation metrics
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.legend()

# Optional: Estimate model's computational complexity (FLOPs)
# Please ensure you have the TensorFlow Profiler installed and available.

def estimate_model_teraflops(model):
     forward_pass = tf.function(model.call, input_signature=[tf.TensorSpec(shape=(1,) + model.input_shape[1:])])
     graph_info = profile(forward_pass.get_concrete_function().graph, options=ProfileOptionBuilder.float_operation())
     flops = graph_info.total_float_ops
     return flops / 10**12

print(estimate_model_teraflops(model))

plt.show()
