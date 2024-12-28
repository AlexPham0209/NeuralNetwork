from numpy import e

def sigmoid(x):
    return 1 / (1 + e ** (-x))