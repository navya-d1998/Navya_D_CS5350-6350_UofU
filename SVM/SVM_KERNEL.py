import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from scipy.optimize import minimize

import os
train_file_path = os.path.abspath('./SVM/bank-note/train.csv')
test_file_path = os.path.abspath('./SVM/bank-note/test.csv')
train_data = pd.read_csv(train_file_path, header=None)
test_data = pd.read_csv(test_file_path, header=None)

# Convert labels to {1, -1}
train_data.iloc[:, -1] = train_data.iloc[:, -1].map({1: 1, 0: -1})
test_data.iloc[:, -1] = test_data.iloc[:, -1].map({1: 1, 0: -1})

# Extract features and labels
X_train = train_data.iloc[:, :-1].values
y_train = train_data.iloc[:, -1].values
X_test = test_data.iloc[:, :-1].values
y_test = test_data.iloc[:, -1].values

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Gaussian kernel function
def gaussian_kernel(x1, x2, gamma):
    return np.exp(-gamma * np.sum((x1 - x2) ** 2))

# Kernel Perceptron algorithm
def kernel_perceptron(X, y, gamma, max_iter=100):
    n_samples, n_features = X.shape
    alpha = np.zeros(n_samples)

    for _ in range(max_iter):
        for i in range(n_samples):
            prediction = np.sum(alpha * y * np.array([gaussian_kernel(X[i], X[j], gamma) for j in range(n_samples)]))
            if y[i] * prediction <= 0:
                alpha[i] += 1

    return alpha

# Kernel Perceptron prediction
def kernel_perceptron_predict(X, X_train, y_train, alpha, gamma):
    n_samples = X.shape[0]
    predictions = np.zeros(n_samples)

    for i in range(n_samples):
        predictions[i] = np.sign(np.sum(alpha * y_train * np.array([gaussian_kernel(X[i], X_train[j], gamma) for j in range(len(X_train))])))

    return predictions

# Nonlinear SVM training function using Gaussian kernel
def train_dual_svm_gaussian(X, y, C, sigma):
    # Get the kernel matrix using the Gaussian kernel.
    n_samples = X.shape[0]
    K = np.zeros((n_samples, n_samples))
    for i in range(n_samples):
        for j in range(n_samples):
            K[i, j] = np.exp(-np.sum((X[i, :] - X[j, :]) ** 2) / (2 * sigma ** 2))

    # Define the dual problem objective function.
    def objective(alpha):
        return 0.5 * np.dot(alpha, np.dot(K, alpha * y) * y) - np.sum(alpha)

    # Define the constraints: alphas must sum to zero.
    constraints = {'type': 'eq', 'fun': lambda alpha: np.sum(alpha * y), 'jac': lambda alpha: y}

    # Define the bounds for alpha: 0 <= alpha_i <= C
    bounds = [(0, C) for _ in range(n_samples)]

    # Solve the dual problem.
    result = minimize(fun=objective,
                      x0=np.zeros(n_samples),
                      method='SLSQP',
                      bounds=bounds,
                      constraints=constraints)

    alphas = result.x
    # Compute the bias term using only the support vectors.
    sv = (alphas > 1e-5)
    b = np.mean(y[sv] - np.dot(K[sv], alphas * y))

    return alphas, b, sv

# Nonlinear SVM prediction function
def svm_predict(X, X_sv, y_sv, alphas_sv, b, gamma):
    # Compute the RBF kernel between X and the support vectors
    K = np.zeros((X.shape[0], X_sv.shape[0]))
    for i in range(X.shape[0]):
        for j in range(X_sv.shape[0]):
            K[i, j] = np.exp(-np.sum((X[i, :] - X_sv[j, :]) ** 2) / (2 * gamma ** 2))
    # Compute the predictions
    predictions = np.dot(K, alphas_sv * y_sv) + b
    return np.sign(predictions)

# Convert DataFrame to NumPy array before passing to the SVM functions
X_train_np = X_train
y_train_np = y_train
X_test_np = X_test
y_test_np = y_test

# Test the Kernel Perceptron with different γ values
gamma_values = [0.1, 0.5, 1, 5, 100]

print("Kernel Perceptron Results:")
for gamma in gamma_values:
    # Train the model
    alpha = kernel_perceptron(X_train, y_train, gamma)

    # Predict on training and test sets
    y_train_pred = kernel_perceptron_predict(X_train, X_train, y_train, alpha, gamma)
    y_test_pred = kernel_perceptron_predict(X_test, X_train, y_train, alpha, gamma)

    # Calculate errors
    train_error = np.mean(y_train_pred != y_train)
    test_error = np.mean(y_test_pred != y_test)

    print(f"Gamma: {gamma}, Training Error Rate: {train_error:.5f}, Testing Error Rate: {test_error:.5f}")

# Test the Nonlinear SVM with Gaussian kernel for comparison
C_values = [100/873, 500/873, 700/873]

print("\nNonlinear SVM Results:")
for C in C_values:
    for gamma in gamma_values:
        # Train the model using NumPy arrays
        alphas, b, sv = train_dual_svm_gaussian(X_train_np, y_train_np, C, gamma)

        # Get the support vectors and their labels
        X_sv = X_train_np[sv]
        y_sv = y_train_np[sv]
        alphas_sv = alphas[sv]

        # Predict on the training and test sets using the support vectors
        y_train_pred = svm_predict(X_train_np, X_sv, y_sv, alphas_sv, b, gamma)
        y_test_pred = svm_predict(X_test_np, X_sv, y_sv, alphas_sv, b, gamma)

        # Calculate errors
        train_error = np.mean(y_train_pred != y_train_np)
        test_error = np.mean(y_test_pred != y_test_np)

        print(f"Gamma: {gamma}, C: {C}, Train Error: {train_error:.5f}, Test Error: {test_error:.5f}")