import numpy as np
import torch
from torch import nn

# this class is used to train networks together in a cascaded approach
# we expect that you have already initialized the networks and put them into a list
class cascaded_networks(nn.Module):
    def __init__(self,
                 networks: list[torch.nn.Module] | nn.ModuleList, # list of networks to chain together
                 input_to_all_networks: bool = True, 
                 transforms_between_networks: list = [],
                 intermediate_outputs: bool = False,
                 split_intermediate_outputs: bool = False
                 ):
        super().__init__()
        """
            networks: list of initialized networks
            input_to_all_networks: boolean parameter that tells us if we should concatenate our
                                    initial input with each network output before passing it to the next network
            transforms_between_networks: list of transforms to apply to network outputs before passing to next network
                                            NOTE: make sure these don't break the computational graph
            intermediate_outputs: boolean that tells us if we should output intermediate outputs
                                    Can be good to calculate loss with more than just the final output if wanted.
        """
        
        # check if our list of networks is already a nn.ModuleList
        if isinstance(networks, nn.ModuleList):
            self.networks = networks
        else: 
            self.networks = torch.nn.ModuleList(networks)
        self.input_skips = input_to_all_networks
        self.data_transforms = transforms_between_networks
        
    def forward(self, input):
        
        # we perform the first network pass outside the loop to keep input untouched in case we want to pass it through the networks
        x = self.networks[0](input)
        # start from 1, passing inputs to next network
        for netIdx in range(1, len(self.networks)):
            # pass to differentiable soft-argmax function
            
            if self.input_skips: # concatenate original input
                x = torch.cat([input, x], dim=0)
            # pass to next network
            x = self.networks[netIdx](x)
            
        # return output
        return x
    
    
    
    def compute_conv_feature_map(self, input_size):
        # iterate through list of networks, calling each's feature map calculation
        feature_map_sum = 0
        for netIdx in range(len(self.networks)):
            feature_map_sum += self.networks[netIdx].compute_conv_feature_map_size(input_size)
                