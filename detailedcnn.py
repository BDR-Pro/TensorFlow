import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import TimeDistributed, Conv2D, MaxPooling2D, Flatten, LSTM, Dense
import matplotlib.pyplot as plt

# Assuming each sequence has 10 images of size 128x128 with 3 color channels (e.g., a video clip)
seq_length = 10  # Number of images in each sequence
img_height = 128
img_width = 128
channels = 3  # RGB channels

# Define the model
model = Sequential()
# TimeDistributed wrapper allows to apply a layer to every temporal slice of an input
model.add(TimeDistributed(Conv2D(32, (3, 3), activation='relu'), input_shape=(seq_length, img_height, img_width, channels)))
model.add(TimeDistributed(MaxPooling2D((2, 2))))
model.add(TimeDistributed(Conv2D(64, (3, 3), activation='relu')))
model.add(TimeDistributed(MaxPooling2D((2, 2))))
model.add(TimeDistributed(Conv2D(128, (3, 3), activation='relu')))
model.add(TimeDistributed(MaxPooling2D((2, 2))))
model.add(TimeDistributed(Flatten()))

# LSTM part
model.add(LSTM(64, return_sequences=False))  # You can stack more LSTM layers if needed

# Fully connected layers
model.add(Dense(64, activation='relu'))
model.add(Dense(10, activation='softmax'))  # Assuming 10 classes for classification

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Model summary
model.summary()


# Import necessary modules
import numpy as np
from tensorflow.keras.utils import to_categorical

# Simulate loading and preprocessing of data
# Here you would load your actual data and preprocess it
num_sequences = 100  # This should match the actual number of sequences you have
train_sequences = np.random.random((num_sequences, seq_length, img_height, img_width, channels))
train_labels = np.random.randint(10, size=(num_sequences,))  # Replace with actual labels

# Convert labels to one-hot encoding
train_labels_one_hot = to_categorical(train_labels, num_classes=10)

# Fit the model on the training data
history = model.fit(train_sequences, train_labels_one_hot, epochs=10, validation_split=0.2)


# Assuming 'history' is the result from the 'model.fit' execution
accuracy = history.history['accuracy']
val_accuracy = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(accuracy) + 1)

# Plot training & validation accuracy values
plt.figure(figsize=(14, 5))
plt.subplot(1, 2, 1)
plt.plot(epochs, accuracy, 'bo-', label='Training accuracy')
plt.plot(epochs, val_accuracy, 'ro-', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

# Plot training & validation loss values
plt.subplot(1, 2, 2)
plt.plot(epochs, loss, 'bo-', label='Training loss')
plt.plot(epochs, val_loss, 'ro-', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
#  Assume 'train_sequences' is your loaded and preprocessed dataset of shape (num_sequences, seq_length, img_height, img_width, channels)
# And 'train_labels' are your labels

#model.fit(train_sequences, train_labels, epochs=10)

from tensorflow.python.profiler.model_analyzer import profile
from tensorflow.python.profiler.option_builder import ProfileOptionBuilder

def estimate_model_teraflops(model):
  forward_pass = tf.function(model.call, input_signature=[tf.TensorSpec(shape=(1,) + model.input_shape[1:])])
  graph_info = profile(forward_pass.get_concrete_function().graph, options=ProfileOptionBuilder.float_operation())
  flops = graph_info.total_float_ops
  return flops  / 10**12


print(estimate_model_teraflops(model))
plt.show()
