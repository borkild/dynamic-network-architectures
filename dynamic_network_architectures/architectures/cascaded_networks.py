import numpy as np
import torch
from torch import nn

from dynamic_network_architectures.building_blocks.soft_argmax import soft_argmax

# this class is used to train networks together in a cascaded approach
# we expect that you have already initialized the networks and put them into a list
class cascaded_networks(nn.Module):
    def __init__(self,
                 networks: list[torch.nn.Module] | nn.ModuleList, # list of networks to chain together
                 deep_supervision: bool = False
                 ):
        super().__init__()
        """
            networks: list of initialized networks
            deep_supervision: boolean to tell us if we will be using deep supervision during training
        """
        
        # check if our list of networks is already a nn.ModuleList
        if isinstance(networks, nn.ModuleList):
            self.networks = networks
        else: 
            self.networks = torch.nn.ModuleList(networks)
        self.deep_supervision = deep_supervision
        # we use a soft argmax function to approximate the argmax normally applied
        self.soft_argmax = soft_argmax()
        
    def forward(self, input):
        seg_outputs = []
        # we perform the first network pass outside the loop to keep input untouched
        x = self.networks[0](input)
        seg_outputs.append(x)
        # start from 1, passing inputs to next network
        for netIdx in range(1, len(self.networks)):
            # pass to differentiable softmax function to get probabilities
            x = torch.softmax(x, 1)
            # get rid of background channel
            x = x[:, 1:]
            # concatenate original input on channel dimension
            x = torch.cat([input, x], dim=1)
            # pass to next network
            x = self.networks[netIdx](x)
            # put into list before softmax operation -- for deep supervision (loss takes logits)
            seg_outputs.append(x)
        
        # return output -- if deep supervision is enabled, then we return list of outputs that includes the intermediate segmentations
        if self.deep_supervision:
            return seg_outputs
        else:
            return seg_outputs[-1]
    
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
                
    
    
if __name__ == "__main__":
    tstInput = torch.rand((1, 1, 128, 128, 128))
    
    from dynamic_network_architectures.architectures.unet import PlainConvUNet
    
    model1 = PlainConvUNet(
    1,
    6,
    (32, 64, 125, 256, 320, 320),
    nn.Conv3d,
    3,
    (1, 2, 2, 2, 2, 2),
    (2, 2, 2, 2, 2, 2),
    3,
    (2, 2, 2, 2, 2),
    False,
    nn.BatchNorm3d,
    None,
    None,
    None,
    nn.ReLU,
    deep_supervision=False,
    )
    
    model2 = PlainConvUNet(
    3,
    6,
    (32, 64, 125, 256, 320, 320),
    nn.Conv3d,
    3,
    (1, 2, 2, 2, 2, 2),
    (2, 2, 2, 2, 2, 2),
    2,
    (2, 2, 2, 2, 2),
    False,
    nn.BatchNorm3d,
    None,
    None,
    None,
    nn.ReLU,
    deep_supervision=False,
    )
    
    tstNet = cascaded_networks([model1, model2])
    
    tstout = tstNet(tstInput)
    
    print("output shape")
    print(tstout.shape)
        
        
        
        
        
        
        