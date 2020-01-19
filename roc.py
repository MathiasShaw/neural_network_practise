import numpy as np 
import matplotlib.pyplot as plt 

def plot(pred, Y): 
    cur = (1.0, 1.0) 
    auc = 0.0 
    n_pos = np.sum(Y == 1) 
    y_step = 1.0 / n_pos 
    x_step = 1.0 / (Y.shape[1] - n_pos) 
    sortedindices = pred.argsort() 
    for i in sortedindices.tolist()[0]: 
        if Y[0][i] == 1: 
            dx = 0
            dy = y_step 
        else:
            dx = x_step 
            dy = 0 
        auc += dx * cur[1]
        plt.plot([cur[0], cur[0] - dx], [cur[1], cur[1] - dy], c = "b") 
        cur = (cur[0] - dx, cur[1] - dy) 

    plt.plot([0, 1], [0, 1], "b--") 
    plt.title("ROC curve") 
    plt.xlabel("FPR") 
    plt.ylabel("TPR") 
    plt.axis([0, 1, 0, 1]) 
    print("AUC: " + str(auc))
    plt.show()