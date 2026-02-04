import numpy as np
import torch
from torch import nn

# this class is for a differentiable softmax function to use between cascaded architectures
class soft_argmax(nn.Module):
    def __init__(self, beta = 12):
        super().__init__()
        self.beta = beta
        
    def forward(self, input):
        pass
    
    
if __name__ == "__main__":
    beta = 12
    y_est = np.array([[1.1, 3.0, 1.1, 1.3, 0.8]])
    a = np.exp(beta*y_est)
    b = np.sum(np.exp(beta*y_est))
    softmax = a/b
    max = np.sum(softmax*y_est)
    print(max)
    pos = range(y_est.size)
    softargmax = np.sum(softmax*pos)
    print(softargmax)