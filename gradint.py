import tensorflow as tf

print(tf.__version__)
print(tf.executing_eagerly())
print(tf.test.is_gpu_available())
print(tf.config.list_physical_devices('GPU'))
print(tf.config.list_physical_devices('CPU'))

# Define parameters
learning_rate = 0.01
epochs = 1000

# Sample data (you can replace these with your actual data)
# For demonstration, we'll use a simple linear relationship: y = 2x + 3
X_train = tf.constant(range(10), dtype=tf.float32)
y_train = tf.constant([2*x + 3 for x in range(10)], dtype=tf.float32)

# Variables for weights and bias
W = tf.Variable(0.0, name='Weight')
b = tf.Variable(0.0, name='Bias')

# Linear regression model
def model(x):
    return W * x + b

# Loss function: Mean squared error
def loss_fn(y_true, y_pred):
    return tf.reduce_mean(tf.square(y_true - y_pred))

# Gradient descent optimizer
optimizer = tf.optimizers.SGD(learning_rate)

# Training process
for epoch in range(epochs):
    with tf.GradientTape() as tape:
        y_pred = model(X_train)
        loss = loss_fn(y_train, y_pred)
    gradients = tape.gradient(loss, [W, b])
    optimizer.apply_gradients(zip(gradients, [W, b]))

    if epoch % 100 == 0:  # Print the loss every 100 epochs
        print(f'Epoch {epoch}: Loss = {loss.numpy()}, W = {W.numpy()}, b = {b.numpy()}')

# Final parameters
print(f'Final parameters - W: {W.numpy()}, b: {b.numpy()}')


