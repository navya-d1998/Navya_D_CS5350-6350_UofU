import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

def initialize_weights(input_size, hidden1_size, hidden2_size, output_size):
    np.random.seed(42)
    weights = {
        'hidden1': np.random.rand(input_size, hidden1_size),
        'hidden2': np.random.rand(hidden1_size, hidden2_size),
        'output': np.random.rand(hidden2_size, output_size)
    }
    return weights

def forward_propagation(X, weights):
    hidden1_input = np.dot(X, weights['hidden1'])
    hidden1_output = sigmoid(hidden1_input)

    hidden2_input = np.dot(hidden1_output, weights['hidden2'])
    hidden2_output = sigmoid(hidden2_input)

    output_input = np.dot(hidden2_output, weights['output'])
    output = sigmoid(output_input)

    return hidden1_output, hidden2_output, output

def backward_propagation(X, y, output, hidden2_output, hidden1_output, weights, learning_rate):
    output_error = y - output
    output_delta = output_error * sigmoid_derivative(output)

    hidden2_error = output_delta.dot(weights['output'].T)
    hidden2_delta = hidden2_error * sigmoid_derivative(hidden2_output)

    hidden1_error = hidden2_delta.dot(weights['hidden2'].T)
    hidden1_delta = hidden1_error * sigmoid_derivative(hidden1_output)

    # Compute gradients for the output layer
    output_gradient = np.outer(hidden2_output, output_delta)

    # Compute gradients for the hidden layers
    hidden2_gradient = np.outer(hidden1_output, hidden2_delta)
    hidden1_gradient = np.outer(X, hidden1_delta)

    # Update weights using gradients and learning rate
    weights['output'] += output_gradient * learning_rate
    weights['hidden2'] += hidden2_gradient * learning_rate
    weights['hidden1'] += hidden1_gradient * learning_rate

def train_neural_network(X_train, y_train, hidden1_size, hidden2_size, output_size, learning_rate, epochs):
    input_size = X_train.shape[1]
    weights = initialize_weights(input_size, hidden1_size, hidden2_size, output_size)

    for epoch in range(epochs):
        for i in range(len(X_train)):
            X = X_train[i]
            y = y_train[i]

            hidden1_output, hidden2_output, output = forward_propagation(X, weights)
            backward_propagation(X, y, output, hidden2_output, hidden1_output, weights, learning_rate)

    return weights

def test_neural_network(X_test, y_test, weights):
    predictions = []
    for i in range(len(X_test)):
        X = X_test[i]
        _, _, output = forward_propagation(X, weights)
        predictions.append(round(output[0]))

    accuracy = np.mean(predictions == y_test)
    return accuracy

# Load data without headers
train_data = pd.read_csv("datasets/bank-note/train.csv", header=None)
test_data = pd.read_csv("datasets/bank-note/test.csv", header=None)

# Extract features and labels
X_train = train_data.iloc[:, :-1].values
y_train = train_data.iloc[:, -1].values
X_test = test_data.iloc[:, :-1].values
y_test = test_data.iloc[:, -1].values

# Normalize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Example usage:
hidden1_size = 4
hidden2_size = 4
output_size = 1
learning_rate = 0.001
epochs = 100

# Train the neural network
trained_weights = train_neural_network(X_train, y_train, hidden1_size, hidden2_size, output_size, learning_rate, epochs)

# Test the neural network
accuracy = test_neural_network(X_test, y_test, trained_weights)

print(f"Accuracy on test set: {accuracy * 100:.2f}%")

