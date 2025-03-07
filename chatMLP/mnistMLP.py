import numpy as np
import pandas as pd
import gzip
import struct
import random
import matplotlib.pyplot as plt

# Load MNIST dataset (from raw IDX format)
def load_mnist(path, kind='train'):
    labels_path = f'{path}/{kind}-labels-idx1-ubyte.gz'
    images_path = f'{path}/{kind}-images-idx3-ubyte.gz'
    
    with gzip.open(labels_path, 'rb') as lbpath:
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8, offset=8)
    
    with gzip.open(images_path, 'rb') as imgpath:
        images = np.frombuffer(imgpath.read(), dtype=np.uint8, offset=16).reshape(len(labels), 784) / 255.0
    
    return images, labels

# Load MNIST data
X_train, y_train = load_mnist('mnist', kind='train')
X_test, y_test = load_mnist('mnist', kind='t10k')

# One-hot encoding for labels
def one_hot_encode(y, num_classes=10):
    return np.eye(num_classes)[y]

y_train_oh = one_hot_encode(y_train)
y_test_oh = one_hot_encode(y_test)

# Initialize network parameters
random.seed(42)
input_size = 784  # 28x28 pixels
hidden_size = 128  # Hidden neurons
output_size = 10   # Digits 0-9

W1 = np.random.randn(input_size, hidden_size) * 0.01
b1 = np.zeros((1, hidden_size))
W2 = np.random.randn(hidden_size, output_size) * 0.01
b2 = np.zeros((1, output_size))

# Activation functions
def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(float)

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))  # Prevent overflow
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

# Cross-entropy loss
def cross_entropy_loss(y_true, y_pred):
    return -np.mean(np.sum(y_true * np.log(y_pred + 1e-8), axis=1))

# Training loop
def train(epochs=1000, learning_rate=0.01, batch_size=64):
    global W1, b1, W2, b2
    
    for epoch in range(epochs):
        indices = np.arange(X_train.shape[0])
        np.random.shuffle(indices)
        X_train_shuffled, y_train_shuffled = X_train[indices], y_train_oh[indices]
        
        total_loss = 0
        
        for i in range(0, X_train.shape[0], batch_size):
            X_batch = X_train_shuffled[i:i+batch_size]
            y_batch = y_train_shuffled[i:i+batch_size]
            
            # Forward propagation
            Z1 = np.dot(X_batch, W1) + b1
            A1 = relu(Z1)
            Z2 = np.dot(A1, W2) + b2
            A2 = softmax(Z2)
            
            # Compute loss
            loss = cross_entropy_loss(y_batch, A2)
            total_loss += loss
            
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
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {total_loss / (X_train.shape[0] // batch_size):.4f}")

# Train the model
train(epochs=100, learning_rate=0.01, batch_size=64)

# Prediction function
def predict(X):
    Z1 = np.dot(X, W1) + b1
    A1 = relu(Z1)
    Z2 = np.dot(A1, W2) + b2
    A2 = softmax(Z2)
    return np.argmax(A2, axis=1)

# Evaluate on test set
y_pred_test = predict(X_test)
accuracy = np.mean(y_pred_test == y_test)
print(f"Test Accuracy: {accuracy:.4f}")
