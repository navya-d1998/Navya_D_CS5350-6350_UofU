import pandas as pd
import numpy as np
import os

train_file_path = os.path.abspath('./Perceptron/bank-note/train.csv')
test_file_path = os.path.abspath('./Perceptron/bank-note/test.csv')

# Load the training and test datasets using Pandas
train_data = pd.read_csv(train_file_path, header=None)
test_data = pd.read_csv(test_file_path, header=None)

# Extract features and labels from the datasets
X_train = train_data.iloc[:, :-1].values  # Features in training data
y_train = train_data.iloc[:, -1].values    # Labels in training data
X_test = test_data.iloc[:, :-1].values     # Features in test data
y_test = test_data.iloc[:, -1].values       # Labels in test data

# Initialize the weight vector and the average weight vector
n = X_train.shape[1]
w = np.zeros(n)
a = np.zeros(n)

# Set the learning rate and the maximum number of epochs
r = 0.1
T = 10

# Training the Averaged Perceptron
for epoch in range(1, T+1):
    for i in range(len(X_train)):
        xi = X_train[i]
        yi = y_train[i]

        if yi * np.dot(w, xi) <= 0:
            w = w + r * yi * xi
        a = a + w

# Calculate the average weight vector
a /= (T * len(X_train))

# Calculate the average prediction error on the test dataset
test_errors = 0
for i in range(len(X_test)):
    if y_test[i] * np.dot(a, X_test[i]) <= 0:
        test_errors += 1

average_error = test_errors / len(X_test)

# Report the learned weight vector (average weight vector) and average prediction error
print("Learned Weight Vector (Average Weight Vector):", a)
print("Average Prediction Error on Test Data:", average_error)