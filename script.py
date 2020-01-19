import numpy as np 
import time
import matplotlib.pyplot as plt
from loadFile import read_file, read_csv
from helper import split_set, shuffle
import neuralNetwork as nn
from plotCurve import learning_curve, learning_rate_curve
import roc 
X, Y, titles = read_csv("data.csv") 
X_norm = nn.normalization(X)
layers_dims = [X.shape[0], 25, 15, 5, 1] 
hyperparameters = nn.hyperparameter_initialization(layers_dims, learning_rate = 0.001, num_iter=20000, print_cost=True, reg_factor= 1)
#learning_rate_curve(X, Y, hyperparameters)
X_shuffled, Y_shuffled = shuffle(X_norm, Y) 
X_train, Y_train, X_test, Y_test = split_set(X_shuffled, Y_shuffled, 0.6)

parameters = nn.model(X_train, Y_train, hyperparameters) 
pred = nn.predict(X_test, parameters) 

roc.plot(pred, Y_test) 
