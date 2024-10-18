import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

sys.path.append('/deci')
from deci import credit_rf_decision as dt  # Use absolute import

# Load train data
columns = ['X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7', 'X8', 'X9', 'X10', 'X11', 'X12', 'X13', 'X14', 'X15', 'X16', 'X17', 'X18', 'X19', 'X20', 'X21', 'X22', 'X23', 'Y']
types = {'X1': int, 'X2': int, 'X3': int, 'X4': int, 'X6': int, 'X7': int, 'X8': int, 'X9': int, 'X10': int, 'X11': int, 'X12': int, 'X13': int, 'X14': int, 'X15': int, 'X16': int, 'X17': int, 'X18': int, 'X19': int, 'X20': int, 'X21': int, 'X22': int, 'X23': int, 'Y': int}


train_file_path = os.path.abspath('./Ensemble Learning/credit/train.csv')
test_file_path = os.path.abspath('./Ensemble Learning/credit/test.csv')


train_data = pd.read_csv(train_file_path, names=columns, dtype=types)
train_size = len(train_data.index)

# Process data: convert numeric to binary
numeric_features = ['X1', 'X5', 'X12', 'X13', 'X14', 'X15', 'X16', 'X17', 'X18', 'X19', 'X20', 'X21', 'X22', 'X23']
for c in numeric_features:
    median = train_data[c].median()
    train_data[c] = train_data[c].apply(lambda x: 0 if x < median else 1)

test_data = pd.read_csv(test_file_path, names=columns, dtype=types)
test_size = len(test_data.index)
for c in numeric_features:
    median = test_data[c].median()
    test_data[c] = test_data[c].apply(lambda x: 0 if x < median else 1)

# Set features and label
features = {'X1': [0, 1], 'X2': [1, 2], 'X3': [0, 1, 2, 3, 4, 5, 6], 'X4': [0, 1, 2, 3], 'X5': [0, 1], 'X6': [-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9], 'X7': [-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9], 'X8': [-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9], 'X9': [-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9], 'X10': [-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9], 'X11': [-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9], 'X12': [0, 1], 'X13': [0, 1], 'X14': [0, 1], 'X15': [0, 1], 'X16': [0, 1], 'X17': [0, 1], 'X18': [0, 1], 'X19': [0, 1], 'X20': [0, 1], 'X21': [0, 1], 'X22': [0, 1], 'X23': [0, 1]}
label = {'Y': [0, 1]}

# Random Forest parameters
T = 500

train_err = [0 for x in range(T)]
test_err = [0 for x in range(T)]
train_predictions = np.array([0 for x in range(train_size)])
test_predictions = np.array([0 for x in range(test_size)])

for t in range(T):
    # Sample with replacement
    sampled = train_data.sample(frac=0.05, replace=True, random_state=t)
    # ID3
    dt_generator = dt.ID3(feature_selection=0, max_depth=17, subset=2)
    # Get decision tree
    decision_tree = dt_generator.generate_decision_tree(sampled, features, label)

    # Predict on train data
    py = dt_generator.classify(decision_tree, train_data) 
    py = np.array(py.tolist())
    py[py == 1] = 1
    py[py == 0] = -1

    train_predictions += py

    py[train_predictions > 0] = 1
    py[train_predictions <= 0] = 0
    train_data['py'] = pd.Series(py)

    acc = train_data.apply(lambda row: 1 if row['Y'] == row['py'] else 0, axis=1).sum() / train_size
    err = 1 - acc
    train_err[t] = err

    # Predict on test data
    py = dt_generator.classify(decision_tree, test_data) 
    py = np.array(py.tolist())
    py[py == 1] = 1
    py[py == 0] = -1

    test_predictions += py

    py[test_predictions > 0] = 1
    py[test_predictions <= 0] = 0
    test_data['py'] = pd.Series(py)
    acc = test_data.apply(lambda row: 1 if row['Y'] == row['py'] else 0, axis=1).sum() / test_size
    err = 1 - acc
    test_err[t] = err
    print('t: ', t, 'train_err: ', train_err[t], 'test_err: ', test_err[t])

# Plot the data
fig, ax = plt.subplots(figsize=(10, 6))

ax.plot(train_err, label='Train Error', color='blue', linestyle='--', linewidth=2)
ax.plot(test_err, label='Test Error', color='orange', linestyle='-.', linewidth=2)
ax.set_xlabel('Iteration', fontsize=14)
ax.set_ylabel('Error Rate', fontsize=14)
ax.legend(loc='upper right', fontsize=12)
ax.set_title('Random Forest, Feature Subset = 2', fontsize=16)
ax.grid(True)

fig.tight_layout()
fig.savefig('rf2_credit.png')
plt.show()

print('done')