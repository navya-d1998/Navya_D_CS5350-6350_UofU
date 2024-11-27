import numpy as np
import pandas as pd
from sklearn.utils import shuffle
import os
print("-"*30)
print("svm")
print("-"*30)  
# Load the data
train_file_path = os.path.abspath('./SVM/bank-note/train.csv')
test_file_path = os.path.abspath('./SVM/bank-note/test.csv')
train_data = pd.read_csv(train_file_path, header=None)
test_data = pd.read_csv(test_file_path, header=None)

# Convert labels to {1, -1}
train_data.iloc[:, -1] = train_data.iloc[:, -1].map({1: 1, 0: -1})
test_data.iloc[:, -1] = test_data.iloc[:, -1].map({1: 1, 0: -1})

# Set the maximum epochs T to 100
max_epochs = 100

# Hyperparameter C values
C_values = [100/873, 500/873, 700/873]

# Learning rate schedule parameters
gamma0 = 0.1
a = 0.01

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
            eta = gamma0 / (1 + (gamma0 / a )* updates)
            margin = y[i] * (np.dot(X[i], w) + b)
            
            if margin < 1:
                w = (1 - eta) * w + eta * C * y[i] * X[i]
                b = b + eta * C * y[i]
            else:
                w = (1 - eta) * w
                
    return w, b

# Training and testing for different hyperparameter settings
for C in C_values:
    X_train, y_train = train_data.iloc[:, :-1].values, train_data.iloc[:, -1].values
    X_test, y_test = test_data.iloc[:, :-1].values, test_data.iloc[:, -1].values
    
    w, b = svm_sgd(X_train, y_train, C, gamma0, a, max_epochs)
    
    # Calculate training error
    train_predictions = np.sign(np.dot(X_train, w) + b)
    train_error = np.mean(train_predictions != y_train)
    
    # Calculate test error
    test_predictions = np.sign(np.dot(X_test, w) + b)
    test_error = np.mean(test_predictions != y_test)
    
    # Print results
    print(f"C: {C}")
    print(f"Training Error: {train_error}, Test Error: {test_error}")
print("-"*30)   
print("primal_svm")  
print("-"*30)

import numpy as np
import pandas as pd
from sklearn.utils import shuffle

# Load the data
train_file_path = os.path.abspath('./SVM/bank-note/train.csv')
test_file_path = os.path.abspath('./SVM/bank-note/test.csv')
train_data = pd.read_csv(train_file_path, header=None)
test_data = pd.read_csv(test_file_path, header=None)
# Convert labels to {1, -1}
train_data.iloc[:, -1] = train_data.iloc[:, -1].map({1: 1, 0: -1})
test_data.iloc[:, -1] = test_data.iloc[:, -1].map({1: 1, 0: -1})

# Set the maximum epochs T to 100
max_epochs = 100

# Hyperparameter C values
C_values = [100/873, 500/873, 700/873]

# Learning rate schedule parameters
gamma0 = 0.1

# SVM with stochastic sub-gradient descent
def svm_sgd(X, y, C, gamma0, max_epochs):
    n, d = X.shape
    w = np.zeros(d)  # Initialize weights to zeros
    b = 0  # Initialize bias to zero
    updates = 0
    
    for epoch in range(max_epochs):
        X, y = shuffle(X, y, random_state=epoch)  # Shuffle data at the start of each epoch
        
        for i in range(n):
            updates += 1
            eta = gamma0 / (1 + updates)
            margin = y[i] * (np.dot(X[i], w) + b)
            
            if margin < 1:
                w = (1 - eta) * w + eta * C * y[i] * X[i]
                b = b + eta * C * y[i]
            else:
                w = (1 - eta) * w
                
    return w, b

# Training and testing for different hyperparameter settings
for C in C_values:
    X_train, y_train = train_data.iloc[:, :-1].values, train_data.iloc[:, -1].values
    X_test, y_test = test_data.iloc[:, :-1].values, test_data.iloc[:, -1].values
    
    w, b = svm_sgd(X_train, y_train, C, gamma0, max_epochs)
    
    # Calculate training error
    train_predictions = np.sign(np.dot(X_train, w) + b)
    train_error = np.mean(train_predictions != y_train)
    
    # Calculate test error
    test_predictions = np.sign(np.dot(X_test, w) + b)
    test_error = np.mean(test_predictions != y_test)
    
    # Print results
    print(f"C: {C}")
    print(f"Training Error: {train_error}, Test Error: {test_error}")
print("-"*30)
print("svm_primalDifference")  
print("-"*30)
import numpy as np
import pandas as pd
from sklearn.utils import shuffle

# Load the data
train_file_path = os.path.abspath('./SVM/bank-note/train.csv')
test_file_path = os.path.abspath('./SVM/bank-note/test.csv')
train_data = pd.read_csv(train_file_path, header=None)
test_data = pd.read_csv(test_file_path, header=None)

# Convert labels to {1, -1}
train_data.iloc[:, -1] = train_data.iloc[:, -1].map({1: 1, 0: -1})
test_data.iloc[:, -1] = test_data.iloc[:, -1].map({1: 1, 0: -1})

# Set the maximum epochs T to 100
max_epochs = 100

# Hyperparameter C values
C_values = [100/873, 500/873, 700/873]

# Learning rate schedule parameters
gamma0_a = 0.1
a = 0.01
gamma0_t = 0.1

# SVM with stochastic sub-gradient descent
def svm_sgd(X, y, C, gamma0, a, max_epochs, schedule_type):
    n, d = X.shape
    w = np.zeros(d)  # Initialize weights to zeros
    b = 0  # Initialize bias to zero
    updates = 0
    
    for epoch in range(max_epochs):
        X, y = shuffle(X, y, random_state=epoch)  # Shuffle data at the start of each epoch
        
        for i in range(n):
            updates += 1
            if schedule_type == 'a':
                eta = gamma0 / (1 + (gamma0 / a) * updates)
            elif schedule_type == 't':
                eta = gamma0 / (1 + updates)
            else:
                raise ValueError("Invalid schedule_type. Use 'a' or 't'.")
            
            margin = y[i] * (np.dot(X[i], w) + b)
            
            if margin < 1:
                w = (1 - eta) * w + eta * C * y[i] * X[i]
                b = b + eta * C * y[i]
            else:
                w = (1 - eta) * w
                
    return w, b

# Training and testing for different hyperparameter settings using gamma0/a schedule
results_a = []
for C in C_values:
    X_train, y_train = train_data.iloc[:, :-1].values, train_data.iloc[:, -1].values
    X_test, y_test = test_data.iloc[:, :-1].values, test_data.iloc[:, -1].values
    
    w, b = svm_sgd(X_train, y_train, C, gamma0_a, a, max_epochs, 'a')
    
    # Calculate training error
    train_predictions = np.sign(np.dot(X_train, w) + b)
    train_error = np.mean(train_predictions != y_train)
    
    # Calculate test error
    test_predictions = np.sign(np.dot(X_test, w) + b)
    test_error = np.mean(test_predictions != y_test)
    
    results_a.append({
        'C': C,
        'w': w,
        'b': b,
        'train_error': train_error,
        'test_error': test_error
    })

# Training and testing for different hyperparameter settings using gamma0/t schedule
results_t = []
for C in C_values:
    X_train, y_train = train_data.iloc[:, :-1].values, train_data.iloc[:, -1].values
    X_test, y_test = test_data.iloc[:, :-1].values, test_data.iloc[:, -1].values
    
    w, b = svm_sgd(X_train, y_train, C, gamma0_t, a, max_epochs, 't')
    
    # Calculate training error
    train_predictions = np.sign(np.dot(X_train, w) + b)
    train_error = np.mean(train_predictions != y_train)
    
    # Calculate test error
    test_predictions = np.sign(np.dot(X_test, w) + b)
    test_error = np.mean(test_predictions != y_test)
    
    results_t.append({
        'C': C,
        'w': w,
        'b': b,
        'train_error': train_error,
        'test_error': test_error
    })

# Compare results between the two learning rate schedules
for res_a, res_t in zip(results_a, results_t):
    print(f"C: {res_a['C']}")

    # Compare differences
    train_error_difference = np.abs(res_a['train_error'] - res_t['train_error'])
    test_error_difference = np.abs(res_a['test_error'] - res_t['test_error'])
    print(f"  Training Error Difference: {train_error_difference}, Test Error Difference: {test_error_difference}")
    
    # Compare model parameters
    weight_difference = np.linalg.norm(res_a['w'] - res_t['w'])
    bias_difference = np.abs(res_a['b'] - res_t['b'])
    print(f"  Weight Difference: {weight_difference}, Bias Difference: {bias_difference}")
    
