import numpy as np
from helper import sigmoid
def initialize_with_zeros(dim):
    W = np.zeros((1, dim))
    b = 0
    return W, b

def forward_propagation(X, parameters):
    W = parameters["W"]
    b = parameters["b"]

    Z = np.dot(W, X) + b
    A = sigmoid(Z)

    return A

def compute_cost(A, Y):
    m = Y.shape[1]
    cost = - 1/m * np.sum(np.sum(Y * np.log(A) + (1-Y) * np.log(1-A)))
    return cost

def backward_propagation(A, X, Y):
    m = Y.shape[1]
    dZ = A - Y
    dW = 1/m * np.dot(dZ, X.T).reshape(Y.shape[0], X.shape[0])
    db = 1/m * np.sum(dZ)
    grads = {"dW": dW,
             "db": db}
    return grads

def update_parameters(parameters, grads, learning_rate):
    parameters["W"] = parameters["W"] - learning_rate * grads["dW"]
    parameters["b"] = parameters["b"] - learning_rate * grads["db"]

def propagate(parameters, X, Y):
    W = parameters["W"]
    b = parameters["b"]
    m = X.shape[1]
    A = sigmoid(np.dot(W, X) + b)
    cost = - 1/m * sum(sum(Y * np.log(A) + (1-Y) * np.log(1-A)))
    dW = np.mean((A-Y) * X, 1)
    db = np.mean(A-Y)
    grads = {"dW": dW,
             "db": db}
    return grads, cost

def optimize(X, Y, parameters, hyperparameters):
    learning_rate = hyperparameters["learning_rate"]
    num_iter = hyperparameters["num_iter"]
    print_cost = hyperparameters["print_cost"]

    costs = []

    for i in range(num_iter):
        A = forward_propagation(X, parameters)
        cost = compute_cost(A, Y)
        grads = backward_propagation(A, X, Y)
        #gradient_check(parameters, grads, X, Y)
        update_parameters(parameters, grads, learning_rate)

        if i % 100 == -1:
            costs.append(cost)
            if print_cost:
                print(parameters)
    return costs

def predict(X, parameters):
    W = parameters["W"]
    b = parameters["b"]
    pred = np.floor(sigmoid(np.dot(W, X) + b) + 0.5)
    return pred

def model(X, Y, hyperparameters):
    W, b = initialize_with_zeros(X.shape[0])
    parameters = {"W": W,
                  "b": b}
    costs = optimize(X, Y, parameters, hyperparameters)
    prediction = predict(X, parameters)
    error = np.mean(np.abs(prediction - Y)) * 100
    print("error: %f%%" %(error))
    return costs, parameters

def gradient_check(parameters, grads, X, Y, epsilon = 1e-7):
    W = parameters["W"]
    b = parameters["b"]
    gradapprox = np.zeros((1, 2))
    grad = grads["dW"]
    for i in range(2):
        thetaplus = np.copy(W)
        thetaplus[0][i] = thetaplus[0][i] + epsilon
        thetaminus = np.copy(W)
        thetaminus[0][i] = thetaminus[0][i] - epsilon
        param_p = {"W": thetaplus, "b": b}
        param_m = {"W": thetaminus, "b": b}
        J_plus = compute_cost(forward_propagation(X, param_p), Y)
        J_minus = compute_cost(forward_propagation(X, param_m), Y)
        gradapprox[0][i] = (J_plus - J_minus) / (2 * epsilon)
    numerator = np.linalg.norm(grad - gradapprox)
    denominator = np.linalg.norm(grad) + np.linalg.norm(gradapprox)
    difference = numerator / denominator
    
    if difference > 2e-7:
        print(difference)
        print("something's wrong")
    return grad