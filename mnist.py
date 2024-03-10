import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.datasets import mnist

# Load and prepare the MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0  # Normalize data to the range [0, 1]

# Build the Sequential model consisting of 3 layers
model = Sequential([
    Flatten(input_shape=(28, 28)),  # Input layer: flattens the 28x28 pixel images
    Dense(128, activation='relu'),  # Hidden layer with 128 neurons and ReLU activation
    Dense(10)                       # Output layer with 10 neurons (one per class)
])

# Choose an optimizer and loss function for training
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
model.compile(optimizer='adam',
              loss=loss_fn,
              metrics=['accuracy'])

# Train and evaluate the model
model.fit(x_train, y_train, epochs=5)
model.evaluate(x_test, y_test, verbose=2)

from tensorflow.python.profiler.model_analyzer import profile
from tensorflow.python.profiler.option_builder import ProfileOptionBuilder

def estimate_model_teraflops(model):
  forward_pass = tf.function(model.call, input_signature=[tf.TensorSpec(shape=(1,) + model.input_shape[1:])])
  graph_info = profile(forward_pass.get_concrete_function().graph, options=ProfileOptionBuilder.float_operation())
  flops = graph_info.total_float_ops
  return flops  / 10**12


print(estimate_model_teraflops(model))