import numpy as np
import torch
from torch import nn

# this class is for a differentiable soft argmax function to use between cascaded architectures
class soft_argmax(nn.Module):
    def __init__(self, beta: float = 1000):
        super().__init__()
        self.beta = beta
        
    def forward(self, input):
        # calculate softmax
        softmax = torch.nn.functional.softmax(self.beta*input, dim=1)
        # get position vector
        pos = torch.arange(start=0, end=input.size(1), step=1)
        # check dimension to get right view adjustment
        if input.dim() == 4:
            pos = pos.view(1, -1, 1, 1)
        elif input.dim() == 5:
            pos = pos.view(1, -1, 1, 1, 1)
        elif input.dim() == 3:
            pos = pos.view(1, -1, 1)
        else:
            raise ValueError("Dimension of input not recognized. Make sure your input is in form B x C x W x H (x D)")
        
        # multiply softmax and position vector, and sum across channels
        softargmax = torch.sum( softmax * pos, dim=1)
        print(softargmax.size())
        
        return softargmax
        
    
    
if __name__ == "__main__":
    input_size = (3,3,2)
    zero_vec = torch.zeros(input_size)
    one_vec = torch.ones(input_size)*.25
    two_vec = torch.randn(input_size)
    combine = torch.stack([zero_vec, one_vec, two_vec], dim=0)
    combine = torch.unsqueeze(combine, dim=0)
    print("input size")
    print(combine.size())
    
    SAM = soft_argmax()
    
    output = SAM(combine)
    
    print("argmax")
    print(torch.argmax(combine, dim=1))
    print("softmax")
    print(output)