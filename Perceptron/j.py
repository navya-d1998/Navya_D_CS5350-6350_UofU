import numpy as np

class Perceptron:
    def __init__(self):
        self.weights = np.ndarray
        
    def __init__(self, X, y, r:float = 1e-3, epochs: int=10):
        self.weights = np.ndarray
        self.train(X, y, r, epochs)

    def append_bias(self, X):
        return np.insert(X, 0, [1]*len(X), axis=1)

    def train(self, X, y, r:float=1e-3, epochs: int=10):
        X = self.append_bias(X)
        self.weights = np.zeros_like(X[0])

        for e in range(epochs):
            idxs = np.arange(len(X))
            np.random.shuffle(idxs)
            for i in idxs:
                if y[i] * np.dot(self.weights, X[i]) <= 0:
                    self.weights += r*(y[i]*X[i])
    
    def predict(self, X) -> np.ndarray:
        X = self.append_bias(X)
        pred = lambda d : np.sign(np.dot(self.weights, d))
        return np.array([pred(xi) for xi in X])

class VotedPerceptron(Perceptron):
    def __init__(self, X, y, r:float = 1e-3, epochs: int=10):
        self.votes = np.ndarray
        self.train(X, y, r, epochs)

    def train(self, X, y, r:float=1e-3, epochs: int=10):
        X = self.append_bias(X)
        m = 0
        weights = [np.zeros_like(X[0])]
        cs = [0]

        for e in range(epochs):
            idxs = np.arange(len(X))
            np.random.shuffle(idxs)
            for i in idxs:
                if y[i] * np.dot(weights[m], X[i]) <= 0:
                    weights[m] += r*(y[i]*X[i])
                    weights.append(weights[m].copy())
                    m += 1
                    cs.append(1)
                else: cs[m] += 1

        self.votes = np.array(list(zip(weights, cs)), dtype=object)
    
    def predict(self, X) -> np.ndarray:
        X = self.append_bias(X)
        preds = np.zeros(len(X), dtype=int)
        for i in range(len(preds)):
            inner = 0
            for w, c in self.votes:
                inner += c * np.sign(np.dot(w, X[i]))
            preds[i] = np.sign(inner)
        return preds

class AveragedPerceptron(Perceptron):
    def train(self, X, y, r:float=1e-3, epochs: int=10):
        X = self.append_bias(X)
        self.weights = np.zeros_like(X[0])
        weights = np.zeros_like(X[0])

        for e in range(epochs):
            idxs = np.arange(len(X))
            np.random.shuffle(idxs)
            for i in idxs:
                if y[i] * np.dot(weights, X[i]) <= 0:
                    weights += r*(y[i]*X[i])
                self.weights = self.weights + weights
                
from os import makedirs
import csv
import numpy as np
import os
np.random.seed(33)

try: makedirs("./out/")
except FileExistsError: None

dataset_loc = "../data/bank-note/"

train_x = []
train_y = []

train_file_path = os.path.abspath('./Perceptron/bank-note/train.csv')
test_file_path = os.path.abspath('./Perceptron/bank-note/test.csv')



with open(train_file_path, "r") as f:
    for line in f:
        terms = line.strip().split(",")
        terms_flt = list(map(lambda x : float(x), terms))
        train_x.append(terms_flt[:-1])
        train_y.append(-1 if terms_flt[-1] == 0 else 1)

train_x = np.array(train_x)
train_y = np.array(train_y)

test_x = []
test_y = []
with open(test_file_path, "r") as f:
    for line in f:
        terms = line.strip().split(",")
        terms_flt = list(map(lambda x : float(x), terms))
        test_x.append(terms_flt[:-1])
        test_y.append(-1 if terms_flt[-1] == 0 else 1)

test_x = np.array(test_x)
test_y = np.array(test_y)

print("==== Standard Perceptron ====")
p = Perceptron(train_x, train_y, r=0.1)
print(f"learned weights: {p.weights}")
print(f"training accuracy: {np.mean(train_y == p.predict(train_x))}")
print(f"testing accuracy: {np.mean(test_y == p.predict(test_x))}")

print("==== Voted Perceptron ====")
vp = VotedPerceptron(train_x, train_y, r=0.1)
print(f"learned weights and counts: {vp.votes}")
print("making csv of weights")
with open('./out/vp_weights.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerow(['b', 'x1', 'x2', 'x3', 'x4', 'Cm'])
    for w in vp.votes:
        row = w[0]
        row = np.append(row, w[1])
        writer.writerow(row)

print(f"training accuracy: {np.mean(train_y == vp.predict(train_x))}")
print(f"testing accuracy: {np.mean(test_y == vp.predict(test_x))}")

print("==== Averaged Perceptron ====")
ap = AveragedPerceptron(train_x, train_y, r=0.1)
print(f"learned weights: {ap.weights}")
print(f"training accuracy: {np.mean(train_y == ap.predict(train_x))}")
print(f"testing accuracy: {np.mean(test_y == ap.predict(test_x))}")