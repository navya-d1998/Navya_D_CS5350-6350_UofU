import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

def initialize_weights_zeros(input_size, hidden_size, output_size):
    weights = {
        'hidden': np.zeros((input_size, hidden_size)),
        'output': np.zeros((hidden_size, output_size))
    }
    return weights

def learning_rate_schedule(gamma_0, d, t):
    return gamma_0 / (1 + (gamma_0 / d) * t)

def forward_propagation(X, weights):
    hidden_input = np.dot(X, weights['hidden'])
    hidden_output = sigmoid(hidden_input)

    output_input = np.dot(hidden_output, weights['output'])
    output = sigmoid(output_input)

    return hidden_output, output

def backward_propagation(X, y, output, hidden_output, weights, learning_rate):
    output_error = y - output
    output_delta = output_error * sigmoid_derivative(output)

    hidden_error = output_delta.dot(weights['output'].T)
    hidden_delta = hidden_error * sigmoid_derivative(hidden_output)

    weights['output'] += np.outer(hidden_output, output_delta) * learning_rate
    weights['hidden'] += np.outer(X, hidden_delta) * learning_rate

def compute_loss(X, y, weights):
    _, output = forward_propagation(X, weights)
    loss = np.mean((y - output) ** 2)
    return loss

def train_neural_network_zeros(X_train, y_train, hidden_size, output_size, gamma_0, d, epochs):
    input_size = X_train.shape[1]
    weights = initialize_weights_zeros(input_size, hidden_size, output_size)
    learning_curve = []

    for epoch in range(epochs):
        # Shuffle training examples
        permutation = np.random.permutation(len(X_train))
        X_train = X_train[permutation]
        y_train = y_train[permutation]

        for i in range(len(X_train)):
            X = X_train[i]
            y = y_train[i]

            gamma_t = learning_rate_schedule(gamma_0, d, epoch * len(X_train) + i)
            
            hidden_output, output = forward_propagation(X, weights)
            backward_propagation(X, y, output, hidden_output, weights, gamma_t)

            # Compute loss for diagnostic purposes
            if i % 100 == 0:
                loss = compute_loss(X_train, y_train, weights)
                learning_curve.append(loss)

    return weights, learning_curve

def test_neural_network_zeros(X_test, y_test, weights):
    _, output = forward_propagation(X_test, weights)
    predictions = np.round(output)
    accuracy = np.mean(predictions == y_test)
    return 1 - accuracy  # Return error instead of accuracy

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

# Settings for hyperparameters
widths = [5, 10, 25, 50, 100]
gamma_0 = 0.5
d = 0.001
epochs = 100

results_zeros = []

for hidden_size in widths:
    print(f"\nTraining with hidden size: {hidden_size} (Weights Initialized to 0)")

    # Train the neural network with weights initialized to 0
    trained_weights_zeros, learning_curve_zeros = train_neural_network_zeros(X_train, y_train, hidden_size, 1, gamma_0, d, epochs)

    # Compute training error
    train_error_zeros = test_neural_network_zeros(X_train, y_train, trained_weights_zeros)

    # Compute test error
    test_error_zeros = test_neural_network_zeros(X_test, y_test, trained_weights_zeros)

    results_zeros.append({
        'hidden_size': hidden_size,
        'train_error': train_error_zeros,
        'test_error': test_error_zeros,
        'learning_curve': learning_curve_zeros
    })

    print(f"Training Error: {train_error_zeros * 100:.2f}% | Test Error: {test_error_zeros * 100:.2f}%")

