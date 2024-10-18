import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class LinearRegression:
    def __init__(self):
        # set default variable: learning rate; stopping criteria; maximal iteration.
        self.lr = 0.0125
        self.stop_criteria = 1e-5
        self.max_iter = 1000
        self.method = 'GD'
    
    # add function to make change for default variables.
    def learning_rate(self, lr):
        self.lr = lr
    
    def set_iteration(self, max_iter):
        self.max_iter = max_iter
    
    def set_threshold(self, stop_criteria):
        self.stop_criteria = stop_criteria
    
    def set_method(self, method):
        self.method = method
    
    # This function computes MSE error.
    def obj_value(self, X, w, y):
        # This X matrix should contains another column of 1 in data processing part, shape mx(n+1)
        # w should be a vector together with weight vector and bias term.(n+1)x1
        # y is a vector with length m, the shape size should be (mx1)
        residual = y - np.matmul(X, w) 
        loss = 0.5 * np.linalg.norm(residual, 2)
        return loss
    
    # This function implements the GD
    def optimizer_gd(self, X, y):
        n = X.shape[1]
        # initialize w
        w = np.zeros((n, 1))
        # will be defined as norm difference as w_t - w_{t+1}
        diff = 10
        count = 0
        fun_val = []
        # loop start
        while diff > self.stop_criteria and count < self.max_iter:
            count += 1
            residual = y - np.matmul(X, w)
            gradient = -np.matmul(np.transpose(X), residual)
            # GD Update
            w_new = w - self.lr * gradient
            diff = np.linalg.norm(w - w_new)
            w = w_new
            # access function value
            current_val = self.obj_value(X, w, y)
            fun_val.append(current_val)
        # make plot
        fig, ax = plt.subplots()
        ax.plot(fun_val, color='blue', linestyle='-', linewidth=2)
        ax.set_title('Gradient Descent', fontsize=16)
        ax.set_xlabel('Iteration', fontsize=14)
        ax.set_ylabel('Loss', fontsize=14)
        ax.legend(['Train by GD'], fontsize=12)
        ax.grid(True)
        fig.tight_layout()
        fig.savefig('grad.png')
        #plt.show()
        return w
    
    # This function implements SGD method
    def optimizer_sgd(self, X, y):
        np.random.seed()
        m = X.shape[0]
        n = X.shape[1]
        # initialize w
        w = np.zeros((n, 1))
        count = 0
        fun_val = []
        
        current_val = 10
        # loop start
        while current_val > self.stop_criteria and count < self.max_iter:
            count += 1 
            # sample an example
            sample_indx = np.random.randint(m, size=1)
            X_row = X[sample_indx, :]
            y_row = y[sample_indx]
            residual = y_row - np.matmul(X_row, w)
            # get stochastic gradient
            sgd = -residual * np.transpose(X_row)
            # sgd update
            w_new = w - self.lr * sgd
            current_val = self.obj_value(X, w_new, y)
            w = w_new
            fun_val.append(current_val)
        # make plot
        fig, ax = plt.subplots()
        ax.plot(fun_val, color='green', linestyle='-', linewidth=2)
        ax.set_title('Stochastic Gradient Descent', fontsize=16)
        ax.set_xlabel('Iteration', fontsize=14)
        ax.set_ylabel('Loss', fontsize=14)
        ax.legend(['Train by SGD'], fontsize=12)
        ax.grid(True)
        fig.tight_layout()
        fig.savefig('sgd.png')
        #plt.show()
        return w
     
    def optimizer_sol(self, X, y):
        # w^* = inv(X^TX)*(X^Ty)
        Xt = np.transpose(X)
        term1 = np.linalg.inv(np.matmul(Xt, X))
        term2 = np.matmul(Xt, y)
        return np.matmul(term1, term2)
        
    # customize optimization method by this function.
    def optimizer(self, X, y):
        if self.method == 'gd':
            return self.optimizer_gd(X, y)
        elif self.method == 'sgd':
            return self.optimizer_sgd(X, y)
        elif self.method == 'optimum':
            return self.optimizer_sol(X, y)