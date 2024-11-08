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

def voted_perceptron(X_train, y_train, T, eta):
    n_samples, n_features = X_train.shape
    w = np.zeros(n_features)  # Initial weight vector
    m = 0  # Initialize m
    distinct_weight_vectors = []  # List to store distinct weight vectors
    correct_counts = []  # List to store the number of correct predictions

    for epoch in range(T):
        mistakes = 0
        for i, x in enumerate(X_train):
            y_pred = np.sign(np.dot(w, x))
            if y_pred == 0:
                y_pred = -1

            if y_pred * y_train[i] <= 0:
                # Update weight vector and m with learning rate
                w += eta * y_train[i] * x
                m += 1
                Cm = 1
                mistakes += 1
            else:
                Cm += 1

        # Store the weight vector and the number of correct predictions for this epoch
        distinct_weight_vectors.append(w.copy())
        correct_counts.append(n_samples - mistakes)

    return distinct_weight_vectors, correct_counts

# Initialize the weight vector, bias, and other variables
n_features = X_train.shape[1]
w = np.zeros(n_features)  # Initial weight vector
b = 0  # Initial bias
learning_rate = 0.1  # You can adjust the learning rate as needed
T = 10  # Maximum number of epochs
distinct_weight_vectors, correct_counts = voted_perceptron(X_train, y_train, T, learning_rate)

# Make predictions on the test dataset using distinct weight vectors
test_errors = []
for w, correct_count in zip(distinct_weight_vectors, correct_counts):
    errors = 0
    for i, x in enumerate(X_test):
        y_pred = np.sign(np.dot(w, x))
        if y_pred == 0:
            y_pred = -1
        if y_pred != y_test[i]:
            errors += 1
    test_errors.append(errors / len(X_test))

# Calculate average test error
average_test_error = np.mean(test_errors)

# Print the distinct weight vectors and their counts
for i, (w, count) in enumerate(zip(distinct_weight_vectors, correct_counts)):
    print(f"Weight Vector {i + 1}: {w},  Count: {count}")

print(f"Average Test Error: {average_test_error:.2f}")



