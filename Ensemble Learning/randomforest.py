import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

sys.path.append('/deci')
from deci import rf_decision as RF  # Use absolute import

# Column names and data types
column_names = ['age', 'job', 'marital', 'education', 'default', 'balance', 'housing', 'loan',
                'contact', 'day', 'month', 'duration', 'campaign', 'pdays', 'previous', 'poutcome', 'y']

data_types = {'age': int, 'job': str, 'marital': str, 'education': str, 'default': str, 'balance': int,
              'housing': str, 'loan': str, 'contact': str, 'day': int, 'month': str, 'duration': int, 'campaign': int,
              'pdays': int, 'previous': int, 'poutcome': str, 'y': str}

# File paths
train_file_path = os.path.abspath('./Ensemble Learning/bank/train.csv')
test_file_path = os.path.abspath('./Ensemble Learning/bank/test.csv')

# Load data
train_df = pd.read_csv(train_file_path, names=column_names, dtype=data_types, header=None)
test_df = pd.read_csv(test_file_path, names=column_names, dtype=data_types, header=None)

train_size = train_df.shape[0]
test_size = test_df.shape[0]

# Preprocess numeric features
numeric_cols = ['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous']
for col in numeric_cols:
    median_train = train_df[col].median()
    train_df[col] = train_df[col].apply(lambda x: 1 if x > median_train else 0).astype(str)
    median_test = test_df[col].median()
    test_df[col] = test_df[col].apply(lambda x: 1 if x > median_test else 0).astype(str)

print(train_df['age'].dtype)
print(train_df['loan'].dtype)
print(test_df['age'].dtype)
print(test_df['loan'].dtype)

# Define features and labels
feature_dict = {'age': ['0', '1'],
                'job': ['admin.', 'unknown', 'unemployed', 'management', 'housemaid', 'entrepreneur', 'student', 'blue-collar', 'self-employed', 'retired', 'technician', 'services'],
                'marital': ['married', 'divorced', 'single'],
                'education': ['unknown', 'secondary', 'primary', 'tertiary'],
                'default': ['yes', 'no'],
                'balance': ['0', '1'],
                'housing': ['yes', 'no'],
                'loan': ['yes', 'no'],
                'contact': ['unknown', 'telephone', 'cellular'],
                'day': ['0', '1'],
                'month': ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec'],
                'duration': ['0', '1'],
                'campaign': ['0', '1'],
                'pdays': ['0', '1'],
                'previous': ['0', '1'],
                'poutcome': ['unknown', 'other', 'failure', 'success']}

label_dict = {'y': ['yes', 'no']}

# Random Forest parameters
num_trees = 500
max_depth = 10
subset_size = 6

train_errors = np.zeros((num_trees,))
test_errors = np.zeros((num_trees,))

train_predictions = np.zeros((train_size,))
test_predictions = np.zeros((test_size,))

metric = 'entropy'
sample_fraction = 0.6
sample_size = round(train_size * sample_fraction)

for tree in range(num_trees):
    sample_indices = np.random.choice(train_size, size=sample_size, replace=True)
    sampled_train_df = train_df.iloc[sample_indices]
    dt_gen = RF.ID3(metric_selection=metric, max_depth=max_depth, attribute_subset=subset_size)
    decision_tree = dt_gen.generate_decision_tree(sampled_train_df, feature_dict, label_dict)
    
    # Train part
    train_pred = dt_gen.classify(decision_tree, train_df)
    train_pred = np.array(train_pred.tolist())
    train_pred[train_pred == 'yes'] = 1
    train_pred[train_pred == 'no'] = -1
    train_pred = train_pred.astype(int)
    train_predictions += train_pred
    
    train_pred = np.where(train_predictions > 0, 'yes', 'no')
    train_df['predicted'] = pd.Series(train_pred)
    
    train_mismatch = train_df.apply(lambda row: 0 if row['y'] == row['predicted'] else 1, axis=1).sum()
    train_errors[tree] = train_mismatch / train_size
    
    # Test part
    test_pred = dt_gen.classify(decision_tree, test_df)
    test_pred = np.array(test_pred.tolist())
    test_pred[test_pred == 'yes'] = 1
    test_pred[test_pred == 'no'] = -1
    test_pred = test_pred.astype(int)
    test_predictions += test_pred
    
    test_pred = np.where(test_predictions > 0, 'yes', 'no')
    test_df['predicted'] = pd.Series(test_pred)
    
    test_mismatch = test_df.apply(lambda row: 0 if row['y'] == row['predicted'] else 1, axis=1).sum()
    test_errors[tree] = test_mismatch / test_size

    print('Iteration:', tree)

# Plot the data
fig, ax = plt.subplots(figsize=(10, 6))

ax.plot(train_errors, label='Train Error', color='blue', linestyle='--', linewidth=2)
ax.plot(test_errors, label='Test Error', color='orange', linestyle='-.', linewidth=2)
ax.set_xlabel('Iteration', fontsize=12)
ax.set_ylabel('Error Rate', fontsize=12)
ax.legend(loc='upper right')
ax.set_title('Training and Test Error for Random Forest Method', fontsize=14)
ax.grid(True)

fig.tight_layout()
fig.savefig('random_forest_error_analysis.png')
plt.show()

print('done')