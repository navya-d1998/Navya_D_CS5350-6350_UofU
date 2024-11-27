import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from scipy.optimize import minimize
import os
train_file_path = os.path.abspath('./SVM/bank-note/train.csv')
test_file_path = os.path.abspath('./SVM/bank-note/test.csv')
train_data = pd.read_csv(train_file_path, header=None)
test_data = pd.read_csv(test_file_path, header=None)

# Convert labels to {1, -1}
train_data.iloc[:, -1] = train_data.iloc[:, -1].map({1: 1, 0: -1})
test_data.iloc[:, -1] = test_data.iloc[:, -1].map({1: 1, 0: -1})

# SVM with stochastic sub-gradient descent
def svm_sgd(X, y, C, gamma0, a, max_epochs):
    n, d = X.shape
    w = np.zeros(d)  # Initialize weights to zeros
    b = 0  # Initialize bias to zero
    updates = 0
    
    for epoch in range(max_epochs):
        X, y = shuffle(X, y, random_state=epoch)  # Shuffle data at the start of each epoch
        
        for i in range(n):
            updates += 1
            eta = gamma0 / (1 + (gamma0 / a) * updates)
            margin = y[i] * (np.dot(X[i], w) + b)
            
            if margin < 1:
                w = (1 - eta) * w + eta * C * y[i] * X[i]
                b = b + eta * C * y[i]
            else:
                w = (1 - eta) * w
    
    return w, b

# SVM in the dual domain
def dual_objective(alpha, X, y, C):
    n, d = X.shape
    w = np.dot(alpha * y, X)
    hinge_loss = np.maximum(1 - np.dot(X, w), 0)
    regularization_term = 0.5 * np.dot(w, w)
    
    dual_objective_value = C * np.sum(hinge_loss) + regularization_term
    
    return dual_objective_value

# Initialize variables for dual SVM
alpha_init = np.zeros(len(train_data))
# Hyperparameter C values for primal SVM
C_values = [100/873, 500/873, 700/873]

primal_results = []
dual_results = []
for C in C_values:
    X_train, y_train = train_data.iloc[:, :-1].values, train_data.iloc[:, -1].values
    
    # Training and testing for different hyperparameter settings using primal SVM
    w_primal, b_primal = svm_sgd(X_train, y_train, C, gamma0=0.5, a=0.01, max_epochs=100)
    
    # Calculate training error for primal SVM
    train_predictions_primal = np.sign(np.dot(X_train, w_primal) + b_primal)
    train_error_primal = np.mean(train_predictions_primal != y_train)
    
    # Calculate test error for primal SVM
    test_predictions_primal = np.sign(np.dot(test_data.iloc[:, :-1].values, w_primal) + b_primal)
    test_error_primal = np.mean(test_predictions_primal != test_data.iloc[:, -1].values)
    
    primal_results.append({
        'C': C,
        'w': w_primal,
        'b': b_primal,
        'train_error': train_error_primal,
        'test_error': test_error_primal
    })

    # Optimize the dual objective function using scipy.optimize.minimize
    result = minimize(dual_objective, alpha_init, args=(X_train, y_train, C), bounds=[(0, C) for _ in range(len(X_train))])
    alpha_optimal = result.x
    
    # Recover feature weights and bias for dual SVM
    w_dual = np.dot(alpha_optimal * y_train, X_train)
    b_dual = y_train - np.dot(X_train, w_dual)
    b_dual = np.mean(b_dual)
    
    # Calculate training error for dual SVM
    train_predictions_dual = np.sign(np.dot(X_train, w_dual) + b_dual)
    train_error_dual = np.mean(train_predictions_dual != y_train)
    
    # Calculate test error for dual SVM
    test_predictions_dual = np.sign(np.dot(test_data.iloc[:, :-1].values, w_dual) + b_dual)
    test_error_dual = np.mean(test_predictions_dual != test_data.iloc[:, -1].values)
    
    dual_results.append({
        'C': C,
        'w': w_dual,
        'b': b_dual,
        'train_error': train_error_dual,
        'test_error': test_error_dual
    })

# Compare results between primal and dual SVM
for res_primal, res_dual in zip(primal_results, dual_results):
    C = res_primal['C']
    w_primal, b_primal = res_primal['w'], res_primal['b']
    w_dual, b_dual = res_dual['w'], res_dual['b']

    # Compare feature weights and bias
    weight_difference = np.linalg.norm(w_primal - w_dual)
    bias_difference = np.abs(b_primal - b_dual)

    # Compare training and test errors
    train_error_difference = np.abs(res_primal['train_error'] - res_dual['train_error'])
    test_error_difference = np.abs(res_primal['test_error'] - res_dual['test_error'])

    # Print results
    print(f"C: {C}")
    print("Primal SVM:")
    print("  w_primal:", w_primal)
    print("  b_primal:", b_primal)
    print("  Train Error: ", res_primal['train_error'])
    print("  Test Error: ", res_primal['test_error'])

    print("Dual SVM:")
    print("  w_dual:", w_dual)
    print("  b_dual:", b_dual)
    print("  Train Error: ", res_dual['train_error'])
    print("  Test Error: ", res_dual['test_error'])

    print("Differences:")
    print("  Feature Weights Difference:", weight_difference)
    print("  Bias Difference:", bias_difference)
    print("  Train Error Difference:", train_error_difference)
    print("  Test Error Difference:", test_error_difference)
    print("-"*30)