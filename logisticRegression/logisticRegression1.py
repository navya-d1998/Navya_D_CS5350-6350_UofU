import numpy as np
import pandas as pd

# Load data without headers
train_data = pd.read_csv("datasets/bank-note/train.csv", header=None)
test_data = pd.read_csv("datasets/bank-note/test.csv", header=None)

# Extract features and labels
X_train = train_data.iloc[:, :-1].values
y_train = train_data.iloc[:, -1].values
X_test = test_data.iloc[:, :-1].values
y_test = test_data.iloc[:, -1].values

# Constants
num_epochs = 100
variances = [0.01, 0.1, 0.5, 1, 3, 5, 10, 100]

# Sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Learning rate schedule
def learning_rate_schedule(gamma_0, d, t):
    return gamma_0 / (1 + (gamma_0 / d) * t)

# Objective function with Gaussian prior
def objective_function(y, y_pred, w, variance):
    likelihood_term = -np.sum(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))
    prior_term = 0.5 / variance * np.sum(w**2)  # Gaussian prior
    return likelihood_term + prior_term

# Gradient of the objective function
def gradient(y, y_pred, X, w, variance):
    return -np.dot(X.T, y - y_pred) + 1 / variance * w

# Training function with Gaussian prior
def train_logistic_regression(X_train, y_train, variance, gamma_0, d, num_epochs):
    input_size = X_train.shape[1]
    w = np.zeros(input_size + 1)  # Including bias term

    for epoch in range(num_epochs):
        permutation = np.random.permutation(len(X_train))
        X_train_permuted = X_train[permutation]
        y_train_permuted = y_train[permutation]

        for i in range(len(X_train_permuted)):
            X = np.concatenate([X_train_permuted[i], [1]])  # Adding bias term
            y = y_train_permuted[i]

            gamma_t = learning_rate_schedule(gamma_0, d, epoch * len(X_train_permuted) + i)

            # Forward pass
            y_pred = sigmoid(np.dot(w, X))
            loss = objective_function(y, y_pred, w, variance)

            # Backward pass
            grad = gradient(y, y_pred, X, w, variance)
            w -= gamma_t * grad

    return w

# Evaluation function
def evaluate(w, X, y):
    X_bias = np.column_stack([X, np.ones(len(X))])  # Adding bias term
    y_pred = sigmoid(np.dot(X_bias, w))
    y_pred_class = (y_pred > 0.5).astype(int)
    error = np.mean(y_pred_class != y)
    return error

# Training and evaluation for different variances
for variance in variances:
    print(f"\nTraining with Variance: {variance}")
    w = train_logistic_regression(X_train, y_train, variance, gamma_0=0.1, d=0.01, num_epochs=num_epochs)

    # Evaluate on training set
    train_error = evaluate(w, X_train, y_train)
    print(f"Training Error: {train_error:.4f}")

    # Evaluate on test set
    test_error = evaluate(w, X_test, y_test)
    print(f"Test Error: {test_error:.4f}")
