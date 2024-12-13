import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import matplotlib.pyplot as plt

# Define the neural network architecture
class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, activation):
        super(NeuralNetwork, self).__init__()
        self.hidden = nn.Linear(input_size, hidden_size)
        if activation == 'tanh':
            nn.init.xavier_uniform_(self.hidden.weight)
        elif activation == 'relu':
            nn.init.kaiming_uniform_(self.hidden.weight, nonlinearity='relu')
        self.activation = nn.Tanh() if activation == 'tanh' else nn.ReLU()
        self.output = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = torch.tensor(x, dtype=torch.float32)  # Convert NumPy array to PyTorch tensor
        x = self.hidden(x)
        x = self.activation(x)
        x = self.output(x)
        return x

    

# Function to compute the learning rate schedule
def learning_rate_schedule(gamma_0, d, t):
    return gamma_0 / (1 + (gamma_0 / d) * t)

# Function to compute the loss
def compute_loss(X, y, model, criterion):
    with torch.no_grad():
        output = model(X)
        loss = criterion(output, y)
    return loss.item()

# Function to train the neural network
def train_neural_network(X_train, y_train, hidden_size, depth, activation, gamma_0, d, epochs, criterion):
    input_size = X_train.shape[1]
    output_size = 1  # Assuming a binary classification problem
    model = NeuralNetwork(input_size, hidden_size, output_size, activation)
    optimizer = optim.Adam(model.parameters(), lr=gamma_0)
    learning_curve = []

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)

    for epoch in range(epochs):
        permutation = torch.randperm(len(X_train_tensor))
        X_train_tensor = X_train_tensor[permutation]
        y_train_tensor = y_train_tensor[permutation]

        for i in range(len(X_train_tensor)):
            X = X_train_tensor[i]
            y = y_train_tensor[i]

            gamma_t = learning_rate_schedule(gamma_0, d, epoch * len(X_train_tensor) + i)
            
            output = model(X)
            loss = criterion(output, y)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Compute loss for diagnostic purposes
            if i % 100 == 0:
                loss = compute_loss(X_train_tensor, y_train_tensor, model, criterion)
                learning_curve.append(loss)

    return model, learning_curve

# Load data without headers
train_data = pd.read_csv("datasets/bank-note/train.csv", header=None)
test_data = pd.read_csv("datasets/bank-note/test.csv", header=None)

# Extract features and labels
X = train_data.iloc[:, :-1].values
y = train_data.iloc[:, -1].values

# Normalize features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Settings for hyperparameters
widths = [5, 10, 25, 50, 100]
depths = [3, 5, 9]
gamma_0 = 1e-3
d = 0.01
epochs = 100

results = []

for activation in ['tanh', 'relu']:
    for hidden_size in widths:
        for depth in depths:
            print(f"\nTraining with activation: {activation}, hidden size: {hidden_size}, depth: {depth}")

            # Train the neural network
            criterion = nn.MSELoss()
            trained_model, learning_curve = train_neural_network(X_train, y_train, hidden_size, depth, activation, gamma_0, d, epochs, criterion)

            # Compute training error
            train_error = compute_loss(X_train, torch.tensor(y_train, dtype=torch.float32), trained_model, criterion)

            # Compute validation error
            val_error = compute_loss(X_val, torch.tensor(y_val, dtype=torch.float32), trained_model, criterion)

            results.append({
                'activation': activation,
                'hidden_size': hidden_size,
                'depth': depth,
                'train_error': train_error,
                'val_error': val_error
            })

            print(f"Training Error: {train_error:.4f} | Validation Error: {val_error:.4f}")

# Print results
print("\nResults:")
for result in results:
    print(f"Activation: {result['activation']}, Hidden Size: {result['hidden_size']}, Depth: {result['depth']}, "
          f"Training Error: {result['train_error']:.4f}, Validation Error: {result['val_error']:.4f}")
