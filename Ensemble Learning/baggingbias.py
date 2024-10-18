import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

sys.path.append('/deci')
from deci import bias_bagging_decision as DT  # Use absolute import

# Load data
column_names = ['age', 'job', 'marital', 'education', 'default', 'balance', 'housing', 'loan',
                'contact', 'day', 'month', 'duration', 'campaign', 'pdays', 'previous', 'poutcome', 'y']

data_types = {'age': int, 'job': str, 'marital': str, 'education': str, 'default': str, 'balance': int,
              'housing': str, 'loan': str, 'contact': str, 'day': int, 'month': str, 'duration': int, 'campaign': int,
              'pdays': int, 'previous': int, 'poutcome': str, 'y': str}

# File paths
train_file_path = os.path.abspath('./Ensemble Learning/bank/train.csv')
test_file_path = os.path.abspath('./Ensemble Learning/bank/test.csv')

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

# Bagging parameters
num_runs = 100
num_trees = 500
max_depth = 10

train_errors = np.zeros((num_trees,))
test_errors = np.zeros((num_trees,))

train_predictions = np.zeros((train_size,))
test_predictions_first_tree = np.zeros((test_size,))
test_predictions = np.zeros((num_runs, test_size))

metric = 'entropy'
sample_size = 1000

for run in range(num_runs):
    sample_indices = np.random.choice(train_size, size=sample_size, replace=False)
    sampled_train_df_round = train_df.iloc[sample_indices]
    sample_train_data_round_size = sampled_train_df_round.shape[0]
    for tree in range(num_trees):
        sample_size_per_tree = 10
        sample_per_tree = np.random.choice(sample_train_data_round_size, size=sample_size_per_tree, replace=True)
        sampled_train_df_round_tree = sampled_train_df_round.iloc[sample_per_tree]
        dt_gen = DT.ID3(metric_selection=metric, max_depth=max_depth)
        decision_tree = dt_gen.generate_decision_tree(sampled_train_df_round_tree, feature_dict, label_dict)
        
        # Test part
        test_pred = dt_gen.classify(decision_tree, test_df)
        test_pred = np.array(test_pred.tolist())
        test_pred[test_pred == 'yes'] = 1
        test_pred[test_pred == 'no'] = -1
        test_pred = test_pred.astype(int)
        test_predictions[run] += test_pred
        if tree == 0:
            test_predictions_first_tree += test_pred
    print('Run:')

ground_truth = np.array(test_df['y'].tolist())
ground_truth[ground_truth == 'yes'] = 1
ground_truth[ground_truth == 'no'] = -1
ground_truth = ground_truth.astype(int)

# 1st tree predictor
test_predictions_first_tree /= num_runs
bias_first_tree = np.mean(np.square(test_predictions_first_tree - ground_truth.astype('float64')))
mean_first_tree = np.mean(test_predictions_first_tree)
variance_first_tree = np.sum(np.square(test_predictions_first_tree - mean_first_tree)) / (test_size - 1)
test_term_first_tree = bias_first_tree + variance_first_tree
print('Bias for first tree is:', bias_first_tree)
print('Variance for first tree is:', variance_first_tree)
print('100 single tree case:', test_term_first_tree)

# Bagged tree predictor
test_predictions_avg = np.sum(test_predictions, axis=0) / (num_runs * num_trees)
bias_bagged_tree = np.mean(np.square(test_predictions_avg - ground_truth.astype('float64')))
mean_bagged_tree = np.mean(test_predictions_avg)
variance_bagged_tree = np.sum(np.square(test_predictions_avg - mean_bagged_tree)) / (test_size - 1)
test_term_bagged_tree = bias_bagged_tree + variance_bagged_tree
print('Bias for bagged tree is:', bias_bagged_tree)
print('Variance for bagged tree is:', variance_bagged_tree)
print('100 bagged tree case:', test_term_bagged_tree)

# Plot the data
fig, ax = plt.subplots(figsize=(10, 6))

ax.plot(train_errors, label='Train Error', color='blue', linestyle='--', linewidth=2)
ax.plot(test_errors, label='Test Error', color='orange', linestyle='-.', linewidth=2)
ax.set_xlabel('Iteration', fontsize=12)
ax.set_ylabel('Error Rate', fontsize=12)
ax.legend(loc='upper right')
ax.set_title('Training and Test Error for Bagging Method', fontsize=14)
ax.grid(True)

fig.tight_layout()
fig.savefig('bagging_error_analysis.png')
plt.show()

print('done')