import numpy as np
import pandas as pd
from scipy.optimize import minimize
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

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)

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
    
    # Use 'L-BFGS-B' as an alternative solver with a higher tolerance
    result = minimize(fun=objective,
                      x0=np.zeros(X.shape[0]),
                      method='L-BFGS-B',
                      bounds=bounds,
                      constraints=constraints,
                      options={'gtol': 1e-6})
    
    alphas = result.x
    # Compute the bias term using only the support vectors.
    sv = (alphas > 1e-5)
    
    return sv

# Convert DataFrame to NumPy array before passing to the SVM functions
X_train_np = X_train
y_train_np = y_train

gamma_values = [0.01, 0.1, 0.5, 1, 5, 100]
C_values = [100/873, 500/873, 700/873]

# Count the number of support vectors for each setting of γ and C
print("Results for counting support vectors:")
for C in C_values:
    for gamma in gamma_values:
        # Train the model using NumPy arrays
        sv = train_dual_svm_gaussian(X_train_np, y_train_np, C, gamma)
        
        # Get the number of support vectors
        num_sv = np.sum(sv)
        
        print(f"Hyperparameter C={C} and gamma={gamma}: Number of Support Vectors: {num_sv}")
    print("-"*35)


# Report the number of overlapped support vectors
print("\nNumber of overlapped support vectors between consecutive gamma values for C=500/873:")
for i in range(len(gamma_values) - 1):
    gamma1 = gamma_values[i]
    gamma2 = gamma_values[i + 1]
    
    # Train the model for consecutive gamma values
    sv1 = train_dual_svm_gaussian(X_train_np, y_train_np, C_values[1], gamma1)
    sv2 = train_dual_svm_gaussian(X_train_np, y_train_np, C_values[1], gamma2)
    
    # Get the indices of overlapped support vectors
    overlapped_sv_indices = np.where(np.logical_and(sv1, sv2))[0]
    num_overlapped_sv = len(overlapped_sv_indices)
    
    print(f"Overlapped support vectors between gamma={gamma1:} and gamma={gamma2:}: {num_overlapped_sv}")