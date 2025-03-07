import numpy as np
import pickle
import random

# Load CIFAR-10 dataset (binary format)
def load_cifar_batch(filename):
    with open(filename, 'rb') as f:
        batch = pickle.load(f, encoding='bytes')
    X = batch[b'data'] / 255.0  # Normalize input
    y = np.array(batch[b'labels'])
    return X, y

# Load dataset
X_train, y_train = load_cifar_batch("data_batch_1")  # Load one batch for simplicity
X_test, y_test = load_cifar_batch("test_batch")

# Convert labels to one-hot encoding
def one_hot_encode(y, num_classes=10):
    one_hot = np.zeros((y.shape[0], num_classes))
    one_hot[np.arange(y.shape[0]), y] = 1
    return one_hot

y_train = one_hot_encode(y_train)
y_test = one_hot_encode(y_test)

# Initialize network parameters
input_size = 3072  # 32x32x3 pixels
hidden_size = 128  # Hidden layer neurons
output_size = 10   # CIFAR-10 has 10 classes
learning_rate = 0.01
batch_size = 100
epochs = 10

W1 = np.random.randn(input_size, hidden_size) * 0.01
b1 = np.zeros((1, hidden_size))
W2 = np.random.randn(hidden_size, output_size) * 0.01
b2 = np.zeros((1, output_size))

# Activation functions
def relu(x):
    return np.maximum(0, x)

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

def relu_derivative(x):
    return (x > 0).astype(float)

# Training loop
for epoch in range(epochs):
    indices = np.arange(len(X_train))
    np.random.shuffle(indices)
    X_train, y_train = X_train[indices], y_train[indices]
    
    for i in range(0, len(X_train), batch_size):
        X_batch = X_train[i:i+batch_size]
        y_batch = y_train[i:i+batch_size]

        # Forward pass
        Z1 = np.dot(X_batch, W1) + b1
        A1 = relu(Z1)
        Z2 = np.dot(A1, W2) + b2
        A2 = softmax(Z2)
        
        # Compute loss
        loss = -np.sum(y_batch * np.log(A2 + 1e-9)) / batch_size
        
        # Backpropagation
        dZ2 = A2 - y_batch
        dW2 = np.dot(A1.T, dZ2) / batch_size
        db2 = np.sum(dZ2, axis=0, keepdims=True) / batch_size
        
        dA1 = np.dot(dZ2, W2.T)
        dZ1 = dA1 * relu_derivative(Z1)
        dW1 = np.dot(X_batch.T, dZ1) / batch_size
        db1 = np.sum(dZ1, axis=0, keepdims=True) / batch_size
        
        # Update weights
        W1 -= learning_rate * dW1
        b1 -= learning_rate * db1
        W2 -= learning_rate * dW2
        b2 -= learning_rate * db2
    
    print(f"Epoch {epoch+1}/{epochs}, Loss: {loss:.4f}")

# Evaluate model
Z1 = np.dot(X_test, W1) + b1
A1 = relu(Z1)
Z2 = np.dot(A1, W2) + b2
A2 = softmax(Z2)
y_pred = np.argmax(A2, axis=1)
y_true = np.argmax(y_test, axis=1)
accuracy = np.mean(y_pred == y_true)
print(f"Test Accuracy: {accuracy * 100:.2f}%")
