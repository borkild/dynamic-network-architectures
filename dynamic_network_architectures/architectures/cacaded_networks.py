import numpy as np
import torch
from torch import nn

from dynamic_network_architectures.building_blocks.soft_argmax import soft_argmax

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
        self.intermediate_outputs = intermediate_outputs
        self.split_intermediates = split_intermediate_outputs
        # we use a soft argmax function to approximate the argmax normally applied
        self.soft_argmax = soft_argmax()
        
    def forward(self, input):
        # we perform the first network pass outside the loop to keep input untouched in case we want to pass it through the networks
        x = self.networks[0](input)
        # start from 1, passing inputs to next network
        for netIdx in range(1, len(self.networks)):
            # save network output size for splitting later
            out_size = x.size()
            # pass to differentiable soft-argmax function
            x = self.soft_argmax(x)
            # check if we need to split our outputs, then concatenate them, so each output becomes its own input channel for the next network
            # note that the background is always skipped, we always assume we aren't going to pass it to the next network
            if self.split_intermediates:
                # create array for previous network output -- take 1 away from number of channels, as we don't use background
                adj_out_size = out_size
                adj_out_size[1] = adj_out_size[1]-1
                x_temp = torch.zeros(adj_out_size)
                for chanIdx in range(1,out_size[1]): # iterate through previous network output channels
                    # we do all this to avoid causing 0 gradients to occur from rounding
                    diff_mat = x - float(chanIdx)
                    diff_mat = torch.round(diff_mat)
                    if input.dim() == 3:
                        x_temp[:, chanIdx-1, :] = x[diff_mat == 0] / float(chanIdx)
                    elif input.dim() == 4:
                        x_temp[:, chanIdx-1, :, :] = x[diff_mat == 0] / float(chanIdx)
                    elif input.dim() == 5:
                        x_temp[:, chanIdx-1, :, :, :] = x[diff_mat == 0] / float(chanIdx)
                    else:
                        raise ValueError("Dimensions are an unrecognized number. We expect input data to be 1D, 2D, or 3D. It should be in the form B x C x W (x H) (x D)")
                # update output with multiple channels seperate, and pixels in that class being ~1 (only around 1 because of soft argmax)  
                x = x_temp
            
            if self.input_skips: # concatenate original input on channel dimension
                x = torch.cat([input, x], dim=1)
            # pass to next network
            x = self.networks[netIdx](x)
            
        # return output
        return x
    
    # function to get the number of output channels from each network in the cascade
    def get_num_output_channels(self):
        output_channels = []
        # for now, we assume nnUnet's basic architectures with an encoder and decoder are being used
        for curNetworkIdx in range(len(self.networks)):
            output_channels.append( self.networks[curNetworkIdx].decoder.numclasses )
            
        return output_channels
            
        
    
    def compute_conv_feature_map(self, input_size):
        # grab number of classes outputted from each network
        numClasses = self.get_num_output_channels()
        # iterate through list of networks, calling each's feature map calculation
        feature_map_sum = 0
        for netIdx in range(len(self.networks)):
            # calculate network memory estimate using the individual network's feature map size function
            feature_map_sum += self.networks[netIdx].compute_conv_feature_map_size(input_size)
            
            # if we're splitting our outputs, then we need to add to the number of channels going to the next network
            if self.split_intermediates:
                temp_input_size = input_size
                temp_input_size[1] = numClasses[0] - 1
            else: # otherwise, we'll just have 1 input channel to the next network
                temp_input_size = input_size
                temp_input_size[1] = 1
            input_size = temp_input_size
            
            # if we are concatenating our original input in later networks, then we need to update the network size
            if self.input_skips:
                input_size[1] += 1
                
        return feature_map_sum
                
                