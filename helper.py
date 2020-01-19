import numpy as np 

def sigmoid(x):
    s = 1 / (1 + np.exp(-x))
    return s

def tanh(x):
    t = (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))
    return t

def relu(x):
    r = np.maximum(0, x)
    return r

def shuffle(X, Y): 
    m = X.shape[1] 
    permutation = list(np.random.permutation(m)) 
    X_shuffled = X[:, permutation] 
    Y_shuffled = Y[:, permutation].reshape((1, m)) 

    return X_shuffled, Y_shuffled 

def split_set(X, Y, portion): 
    X_shuffle, Y_shuffle = shuffle(X, Y) 
    m = X_shuffle.shape[1] 
    line = np.ceil(portion * m).astype(int)
    X_1 = X_shuffle[:, 0:line] 
    Y_1 = Y_shuffle[:, 0:line] 
    X_2 = X_shuffle[:, line:m+1]
    Y_2 = Y_shuffle[:, line:m+1]

    return X_1, Y_1, X_2, Y_2 