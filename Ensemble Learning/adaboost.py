import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import ExternalFunctions as WDT

# Load and preprocess data
def load_dataset(train_file_path, test_file_path, col_names, data_types):
    train_df = pd.read_csv(train_file_path, names=col_names, dtype=data_types, header=None)
    test_df = pd.read_csv(test_file_path, names=col_names, dtype=data_types, header=None)
    return train_df, test_df

def preprocess_dataset(df, numeric_cols):
    for col in numeric_cols:
        median_val = df[col].median()
        df[col] = df[col].apply(lambda x: 1 if x > median_val else 0).astype(str)
    return df

# Define features and labels
col_names = ['age', 'job', 'marital', 'education', 'default', 'balance', 'housing', 'loan',
             'contact', 'day', 'month', 'duration', 'campaign', 'pdays', 'previous', 'poutcome', 'y']

data_types = {'age': int, 'job': str, 'marital': str, 'education': str, 'default': str, 'balance': int,
              'housing': str, 'loan': str, 'contact': str, 'day': int, 'month': str, 'duration': int,
              'campaign': int, 'pdays': int, 'previous': int, 'poutcome': str, 'y': str}

numeric_cols = ['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous']

feature_dict = {'age': ['0', '1'], 'job': ['admin.', 'unknown', 'unemployed', 'management', 'housemaid', 'entrepreneur',
                'student', 'blue-collar', 'self-employed', 'retired', 'technician', 'services'],
                'marital': ['married', 'divorced', 'single'], 'education': ['unknown', 'secondary', 'primary', 'tertiary'],
                'default': ['yes', 'no'], 'balance': ['0', '1'], 'housing': ['yes', 'no'], 'loan': ['yes', 'no'],
                'contact': ['unknown', 'telephone', 'cellular'], 'day': ['0', '1'], 'month': ['jan', 'feb', 'mar', 'apr',
                'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec'], 'duration': ['0', '1'], 'campaign': ['0', '1'],
                'pdays': ['0', '1'], 'previous': ['0', '1'], 'poutcome': ['unknown', 'other', 'failure', 'success']}

label_dict = {'y': ['yes', 'no']}

# Paths to data files
train_file_path = os.path.abspath('./Ensemble Learning/bank/train.csv')
test_file_path = os.path.abspath('./Ensemble Learning/bank/test.csv')

# Load and preprocess data
train_df, test_df = load_dataset(train_file_path, test_file_path, col_names, data_types)
train_df = preprocess_dataset(train_df, numeric_cols)
test_df = preprocess_dataset(test_df, numeric_cols)

train_size = train_df.shape[0]
test_size = test_df.shape[0]

# Initialize variables for AdaBoost
num_iterations = 100
alpha_vals = np.zeros((num_iterations,))
sample_weights = np.ones(train_size) / train_size
train_errors = np.zeros((num_iterations,))
test_errors = np.zeros((num_iterations,))
cumulative_train_errors = np.zeros((num_iterations,))
cumulative_test_errors = np.zeros((num_iterations,))
train_predictions = np.zeros((train_size,))
test_predictions = np.zeros((test_size,))

# AdaBoost algorithm
for i in range(num_iterations):
    dt_gen = WDT.WeightedID3(metric_selection='entropy', max_depth=2)
    decision_tree = dt_gen.generate_decision_tree(train_df, feature_dict, label_dict, sample_weights)
    
    train_df['predicted'] = dt_gen.classify(decision_tree, train_df)
    incorrect_predictions = train_df.apply(lambda row: 0 if row['y'] == row['predicted'] else 1, axis=1)
    error_rate = incorrect_predictions.sum() / train_size
    train_errors[i] = error_rate
    
    incorrect_predictions = train_df.apply(lambda row: 1 if row['y'] == row['predicted'] else -1, axis=1)
    incorrect_predictions = np.array(incorrect_predictions.tolist())
    error_rate = np.sum(sample_weights[incorrect_predictions == -1])
    
    alpha = 0.5 * np.log((1 - error_rate) / error_rate)
    alpha_vals[i] = alpha
    sample_weights *= np.exp(-alpha * incorrect_predictions)
    sample_weights /= np.sum(sample_weights)
    
    test_df['predicted'] = dt_gen.classify(decision_tree, test_df)
    incorrect_predictions = test_df.apply(lambda row: 0 if row['y'] == row['predicted'] else 1, axis=1)
    test_errors[i] = incorrect_predictions.sum() / test_size
    
    train_pred = np.array(train_df['predicted'].tolist())
    train_pred[train_pred == 'yes'] = 1
    train_pred[train_pred == 'no'] = -1
    train_pred = train_pred.astype(int)
    train_predictions += alpha * train_pred
    train_pred = np.where(train_predictions > 0, 'yes', 'no')
    train_df['predicted'] = pd.Series(train_pred)
    cumulative_train_errors[i] = train_df.apply(lambda row: 0 if row['y'] == row['predicted'] else 1, axis=1).sum() / train_size
    
    test_pred = np.array(test_df['predicted'].tolist())
    test_pred[test_pred == 'yes'] = 1
    test_pred[test_pred == 'no'] = -1
    test_pred = test_pred.astype(int)
    test_predictions += alpha * test_pred
    test_pred = np.where(test_predictions > 0, 'yes', 'no')
    test_df['predicted'] = pd.Series(test_pred)
    cumulative_test_errors[i] = test_df.apply(lambda row: 0 if row['y'] == row['predicted'] else 1, axis=1).sum() / test_size

# Plot results
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

ax1.plot(train_errors, 'b--', linewidth=2, label='Train Error')
ax1.plot(test_errors, 'r-.', linewidth=2, label='Test Error')
ax1.legend(loc='upper right')
ax1.set_title('Individual Tree Errors')
ax1.set_xlabel('Iteration')
ax1.set_ylabel('Error Rate')
ax1.grid(True)

ax2.plot(cumulative_train_errors, 'b-', linewidth=2, label='Cumulative Train Error')
ax2.plot(cumulative_test_errors, 'r-', linewidth=2, label='Cumulative Test Error')
ax2.legend(loc='upper right')
ax2.set_title('Cumulative Errors')
ax2.set_xlabel('Iteration')
ax2.set_ylabel('Error Rate')
ax2.grid(True)

fig.suptitle('AdaBoost Error Analysis', fontsize=16)
fig.tight_layout(rect=[0, 0.03, 1, 0.95])
fig.savefig('adaboost_error_analysis.png')
plt.show()

print('done')