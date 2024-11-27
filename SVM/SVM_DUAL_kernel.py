import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

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

# Define the Gaussian kernel matrix function
def gaussian_kernel_matrix(X, sigma):
    n_samples = X.shape[0]
    K = np.zeros((n_samples, n_samples))
    for i in range(n_samples):
        for j in range(n_samples):
            K[i, j] = np.exp(-np.sum((X[i, :] - X[j, :]) ** 2) / (2 * sigma ** 2))
    return K

# Define the dual SVM training function using the Gaussian kernel
def train_dual_svm_gaussian(X, y, C, sigma):
    # Get the kernel matrix using the Gaussian kernel.
    K = gaussian_kernel_matrix(X, sigma)
    
    # Define the dual problem objective function.
    def objective(alpha):
        return 0.5 * np.dot(alpha, np.dot(K, alpha * y) * y) - np.sum(alpha)
    
    # Define the constraints: alphas must sum to zero.
    constraints = {'type': 'eq', 'fun': lambda alpha: np.sum(alpha * y), 'jac': lambda alpha: y}
    
    # Define the bounds for alpha: 0 <= alpha_i <= C
    bounds = [(0, C) for _ in range(X.shape[0])]
    
    # Solve the dual problem.
    result = minimize(fun=objective,
                      x0=np.zeros(X.shape[0]),
                      method='SLSQP',
                      bounds=bounds,
                      constraints=constraints,
                      options={'ftol': 1e-10, 'disp': False})
    
    alphas = result.x
    # Compute the bias term using only the support vectors.
    sv = (alphas > 1e-5)
    b = np.mean(y[sv] - np.dot(K[sv], alphas * y))
    
    return alphas, b, sv

# SVM prediction function
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

gamma_values = [0.1, 0.5, 1, 5, 100]
C_values = [100/873, 500/873, 700/873]

# Training and prediction loop
best_error = float('inf')
best_params = None

for gamma in gamma_values:
    for C in C_values:
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
        
        print("------------------------------------------------------------")
        print(f"Evaluation for C={C:.5f} and gamma={gamma:.5f}:")
        print(f"Training Error Rate: {train_error:.5f}")
        print(f"Testing Error Rate: {test_error:.5f}")
        print("------------------------------------------------------------")

        # Update best parameters if needed
        if test_error < best_error:
            best_error = test_error
            best_params = {'gamma': gamma, 'C': C}

print(f"\nBest Parameters: {best_params}, Best Test Error: {best_error:.5f}")
