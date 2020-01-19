import numpy as np 

def initialization(layers_dims): 
    V = {} 
    S = {} 
    L = len(layers_dims) 
    for l in range(1, L): 
        V["dW" + str(l)] = np.zeros((layers_dims[l], layers_dims[l-1]))
        S["dW" + str(l)] = np.zeros((layers_dims[l], layers_dims[l-1]))
        V["db" + str(l)] = np.zeros((layers_dims[l], 1))
        S["db" + str(l)] = np.zeros((layers_dims[l], 1))
    adam_param = (V, S) 
    return adam_param

def update(t, grads, adam_param, beta1 = 0.9, beta2 = 0.999, epsilon = 1e-8): 
    V, S = adam_param 
    L = len(V) // 2
    adam_descent = {}
    for l in range(1, L+1): 
        V["dW" + str(l)] = beta1 * V["dW" + str(l)] + (1-beta1) * grads["dW" + str(l)] 
        V["db" + str(l)] = beta1 * V["db" + str(l)] + (1-beta1) * grads["db" + str(l)] 
        S["dW" + str(l)] = beta2 * S["dW" + str(l)] + (1-beta2) * np.square(grads["dW" + str(l)])
        S["db" + str(l)] = beta2 * S["db" + str(l)] + (1-beta2) * np.square(grads["db" + str(l)]) 

        vc_corr = 1 - np.power(beta1, t)
        sc_corr = 1 - np.power(beta2, t) 
        VdWc = V["dW" + str(l)] / vc_corr
        Vdbc = V["db" + str(l)] / vc_corr 
        SdWc = S["dW" + str(l)] / sc_corr 
        Sdbc = S["db" + str(l)] / sc_corr 

        adam_descent["dW" + str(l)] = VdWc / (np.sqrt(SdWc) + epsilon)
        adam_descent["db" + str(l)] = Vdbc / (np.sqrt(Sdbc) + epsilon) 
    return adam_descent  

