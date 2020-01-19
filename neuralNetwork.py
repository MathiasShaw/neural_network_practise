import numpy as np 
import matplotlib.pyplot as plt 
import adam
from helper import sigmoid, relu, split_set

def random_initialization(layers_dims): 
    parameters = {}
    L = len(layers_dims)
    for l in range(1, L): 
        parameters["W" + str(l)] = np.random.randn(layers_dims[l], layers_dims[l-1]) * np.sqrt(2/layers_dims[l-1])
        parameters["b" + str(l)] = np.zeros((layers_dims[l], 1))
    return parameters

def hyperparameter_initialization(layers_dims, learning_rate = 0.002, reg_factor = 0.001, num_iter = 20000, print_cost = True): 
    hyperparameter = {"layers_dims": layers_dims,
                      "learning_rate": learning_rate,
                      "reg_factor": reg_factor,
                      "num_iter": num_iter, 
                      "print_cost": print_cost}
    return hyperparameter 

def normalization(X): 
    mu = np.mean(X, 1, keepdims = True)
    sigma = np.std(X, 1, keepdims = True)
    X_norm = (X - mu) / sigma
    return X_norm 


def linear_activation_forward(A_prev, W, b, activation): 
    Z = np.dot(W, A_prev) + b 
    if activation == "relu":
        A = relu(Z) 
    elif activation == "sigmoid":
        A = sigmoid(Z) 
    cache = (A_prev, Z, W, b) 
    return A, cache 

def forward_propagation(X, parameters): 
    L = len(parameters) // 2
    A_prev = X 
    caches = []
    for l in range(1, L): 
        A, cache = linear_activation_forward(A_prev, parameters["W" + str(l)], parameters["b" + str(l)], "relu") 
        caches.append(cache) 
        A_prev = A 
    AL, cache = linear_activation_forward(A_prev, parameters["W" + str(L)], parameters["b" + str(L)], "sigmoid") 
    caches.append(cache) 
    AL = AL + (0.5-AL) * 1e-10 
    return AL, caches 

def compute_cost(AL, Y): 
    cost = -np.mean(Y * np.log(AL) + (1-Y) * np.log(1-AL))
    cost = np.squeeze(cost) 
    return cost 

def reg_cost(cost, parameters): 
    L = len(parameters) // 2
    for l in range(1, L+1): 
        cost += np.sum(np.sum(np.square(parameters["W" + str(l)])))
    return cost 

def linear_activation_backward(dA, cache, reg_factor, activation): 
    A_prev, Z, W, _ = cache 
    if activation == "relu": 
        dZ = dA * (Z > 0).astype(int) 
    elif activation == "sigmoid": 
        dZ = dA 
    m = A_prev.shape[1] 
    dW = 1/m * np.dot(dZ, A_prev.T) + reg_factor/m * W 
    db = np.mean(dZ, axis = 1, keepdims= True) 
    dA_prev = np.dot(W.T, dZ) 

    return dA_prev, dW, db 

def backward_propagation(AL, Y, caches, reg_factor): 
    grads = {} 
    L = len(caches) 

    #dAL = - (np.divide(Y, AL) - np.divide(1-Y, 1-AL))
    dAL = AL - Y
    grads["dA" + str(L-1)], grads["dW" + str(L)], grads["db" + str(L)] = linear_activation_backward(dAL, caches[L-1], 1, "sigmoid") 
    for l in reversed(range(L-1)): 
        dA_prev, dW, db = linear_activation_backward(grads["dA" + str(l+1)], caches[l], reg_factor, "relu") 
        grads["dW" + str(l+1)] = dW 
        grads["db" + str(l+1)] = db 
        grads["dA" + str(l)] = dA_prev 
    return grads 


def update_parameters_adam(t, parameters, grads, adam_param, learning_rate): 
    L = len(parameters) // 2
    learning_rate = 1 / (1 + 0.0001 * t) * learning_rate
    adam_descent = adam.update(t, grads, adam_param) 
    for l in range(1, L+1): 
        parameters["W" + str(l)] -= learning_rate * adam_descent["dW" + str(l)]
        parameters["b" + str(l)] -= learning_rate * adam_descent["db" + str(l)]

def optimization(X, Y, parameters, adam_param, hyperparameters): 
    learning_rate = hyperparameters["learning_rate"] 
    reg_factor = hyperparameters["reg_factor"] 
    num_iter = hyperparameters["num_iter"] 
    print_cost = hyperparameters["print_cost"] 

    costs = []
    for i in range(num_iter): 
        AL, caches = forward_propagation(X, parameters) 
        cost = compute_cost(AL, Y) 
        grads = backward_propagation(AL, Y, caches, reg_factor) 
        update_parameters_adam(i+1, parameters, grads, adam_param, learning_rate)

        if print_cost and i % 100 == 0: 
            print("\rcost after %i: %f" %(i, cost), end='', flush=True)

        costs.append(cost) 
    print("\r")
    return costs[-1]

def predict(X, parameters, threshold = 0.5): 
    AL, _ = forward_propagation(X, parameters) 
    pred = np.floor(AL + threshold) 
    return pred 

def compute_accuracy(pred, Y): 
    err = np.sum(np.sum(np.abs(Y-pred)))
    err /= Y.shape[1] 
    return 1-err, err 


def model(X, Y, hyperparameters): 
    layers_dims = hyperparameters["layers_dims"] 
    parameters = random_initialization(layers_dims) 
    adam_param = adam.initialization(layers_dims) 
    optimization(X, Y, parameters, adam_param, hyperparameters) 
    _, train_error = compute_accuracy(predict(X, parameters), Y) 
    print("train error: %f%%" %(train_error*100))
    return parameters

def plot_decision_boundary(X, Y, parameters, cmap = "RdBu"): 
    cmap = plt.get_cmap(cmap) 

    xmin, xmax = np.min(X[0, :]) - 1, np.max(X[0, :]) + 1
    ymin, ymax = np.min(X[1, :]) - 1, np.max(X[1, :]) + 1 
    steps = 1000 
    x_span = np.linspace(xmin, xmax, steps) 
    y_span = np.linspace(ymin, ymax, steps) 
    xx, yy = np.meshgrid(x_span, y_span)
    x_test = np.array([xx.reshape(1000000), yy.reshape(1000000)])

    y_hat = predict(x_test, parameters) 
    y_hat = y_hat.reshape(xx.shape)
    
    plt.pcolormesh(xx, yy, y_hat, cmap = "PiYG") 
    plt.scatter(X[0, :], X[1, :], Y.T, cmap = "PuOr")
    plt.show()