import numpy as np 
import matplotlib.pyplot as plt 
import neuralNetwork as nn 
from helper import split_set
def learning_curve(X, Y, hyperparameters): 
    X_train, Y_train, X_test, Y_test = split_set(X, Y, 0.6) 
    train_error = []
    test_error = [] 
    n, m = X_train.shape
    layers_dims = hyperparameters["layers_dims"] 
    X_sub = np.array([]).reshape((n, 0))
    Y_sub = np.array([]).reshape((1, 0))
    for i in range(m): 
        parameters = nn.random_initialization(layers_dims)
        adam_param = nn.adam.initialization(layers_dims)
        X_sub = np.c_[X_sub, X_train[:, i]]
        Y_sub = np.c_[Y_sub, Y_train[:, i]]
        nn.optimization(X_sub, Y_sub, parameters, adam_param, hyperparameters)
        _, train_err = nn.compute_accuracy(nn.predict(X_train, parameters), Y_train)
        _, test_err = nn.compute_accuracy(nn.predict(X_test, parameters), Y_test)
        train_error.append(train_err)
        test_error.append(test_err)

    
    plt.plot(list(range(1, m+1)), train_error, c = "b")
    plt.plot(list(range(1, m+1)), test_error, c = "r")
    plt.xlim(0, m+10) 
    plt.ylim(0, 1)
    plt.show()

def learning_rate_curve(X, Y, hyperparameters): 
    lr_vec = np.arange(0.001, 0.02, 0.001)
    X_norm = nn.normalization(X)
    X_train, Y_train, X_test, Y_test = split_set(X_norm, Y, 0.6) 
    train_error = []
    test_error = [] 
    layers_dims = hyperparameters["layers_dims"] 
    
    test_errors = []
    train_errors = []
    for i in range(len(lr_vec)): 
        hyperparameters["learning_rate"] = lr_vec[i] 
        parameters = nn.random_initialization(layers_dims) 
        adam_param = nn.adam.initialization(layers_dims) 
        nn.optimization(X_train, Y_train, parameters, adam_param, hyperparameters) 
        _, train_error = nn.compute_accuracy(nn.predict(X_train, parameters), Y_train) 
        _, test_error = nn.compute_accuracy(nn.predict(X_test, parameters), Y_test) 
        train_errors.append(train_error) 
        test_errors.append(test_error) 
    
    plt.plot(lr_vec, train_errors, c = "b") 
    plt.plot(lr_vec, test_errors, c = "r") 
    plt.xlim(0, 3) 
    plt.ylim(0, 1) 
    plt.show()

