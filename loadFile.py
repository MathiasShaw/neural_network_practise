import numpy as np

def read_file(filename, separator = " "):
    content = []
    with open(filename, "r") as file_to_load:
        while True:
            line = file_to_load.readline()
            if not line:
                break
            content_tmp = [float(i) for i in line.split(separator)]
            content.append(content_tmp)
        content = np.array(content)
    X = np.array([[]])
    Y = np.array([[]])
    if content != []:
        X = content[:, 0:-1].T
        Y = content[:, -1].reshape(1, X.shape[1])
    return X, Y

def read_csv(filename): 
    content = [] 
    titles = [] 
    with open(filename, "r") as file_to_load: 
        titles = file_to_load.readline().split(",") 
        while True: 
            line = file_to_load.readline() 
            if not line: 
                break 
            content.append([float(i) for i in line.split(",")]) 
        content = np.array(content) 
    X = np.array([[]]) 
    Y = np.array([[]]) 
    titles.pop(-1)
    if content != []: 
        X = content[:, 0:-1].T 
        Y = content[:, -1].T.reshape(1, X.shape[1])
    return X, Y, titles 