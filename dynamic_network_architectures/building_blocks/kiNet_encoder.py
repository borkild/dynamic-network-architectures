import torch
from torch import nn
import numpy as np
from typing import Type

from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.dropout import _DropoutNd
from dynamic_network_architectures.building_blocks.simple_conv_blocks import StackedConvBlocks
from dynamic_network_architectures.building_blocks.helper import maybe_convert_scalar_to_list

# simple encoder that does upsamples with each resolution change for KiNet
class PlainUpsampleEncoder(torch.nn.Module):
    def __init__(self,
                 input_channels: int,
                 n_stages: int, # network depth
                 features_per_stage: int | list[int] | tuple[int, ...],
                 conv_op: Type[_ConvNd],
                 kernel_sizes: int | list[int] | tuple[int, ...],
                 strides: int | list[int] | tuple[int, ...],
                 n_conv_per_stage: int | list[int] | tuple[int, ...],
                 interpScaleFactors: int | list[int] | tuple[int, ...],
                 interpMode: str,
                 conv_bias: bool = False,
                 norm_op: None | Type[nn.Module] = None,
                 norm_op_kwargs: dict = None,
                 dropout_op: None | Type[_DropoutNd] = None,
                 dropout_op_kwargs: dict = None,
                 nonlin: None | Type[torch.nn.Module] = None,
                 nonlin_kwargs: dict = None,
                 nonlin_first: bool = False,
                 return_skips: bool = False ):
                super().__init__()
                # check input types and convert to list of correct size if given an int
                if isinstance(kernel_sizes, int):
                    kernel_sizes = [kernel_sizes] * n_stages
                if isinstance(features_per_stage, int):
                    features_per_stage = [features_per_stage] * n_stages
                if isinstance(n_conv_per_stage, int):
                    n_conv_per_stage = [n_conv_per_stage] * n_stages
                if isinstance(strides, int):
                    strides = [strides] * n_stages
                if isinstance(interpScaleFactors, int):
                    interpScaleFactors = [interpScaleFactors] * n_stages
                assert len(kernel_sizes) == n_stages, "kernel_sizes must have as many entries as we have resolution stages (n_stages)"
                assert len(n_conv_per_stage) == n_stages, "n_conv_per_stage must have as many entries as we have resolution stages (n_stages)"
                assert len(features_per_stage) == n_stages, "features_per_stage must have as many entries as we have resolution stages (n_stages)"
                assert len(strides) == n_stages, "strides must have as many entries as we have resolution stages (n_stages). " \
                                                    "Important: first entry is recommended to be 1, else we run strided conv drectly on the input"
                assert len(interpScaleFactors) == n_stages, "scale factors must have as may entries as we have resolution stages (n_stages)."
                
                # iterate through network stages, building out convolution blocks with upsamples
                stages = []
                for curStage in range(n_stages):
                    stageLayers = [] # empty list for this stage's layers
                    # conv layers first
                    stageLayers.append( StackedConvBlocks(n_conv_per_stage[curStage], conv_op, input_channels,
                                                          features_per_stage[curStage], kernel_sizes[curStage], strides[curStage], 
                                                          conv_bias, norm_op, norm_op_kwargs, dropout_op, dropout_op_kwargs,
                                                          nonlin, nonlin_kwargs, nonlin_first) )
                    # upsample after conv layers -- technically reverses order from KiUnet paper, but follows original Unet order -- shouldn't effect output
                    stageLayers.append( interpolate(scaleFactor=interpScaleFactors[curStage], interpMode=interpMode) )
                    stages.append( nn.Sequential(*stageLayers) )
                    input_channels = features_per_stage[curStage] # update input channels for next iteration
                    
                self.stages = nn.Sequential(*stages)
                self.outputChannels = features_per_stage
                self.strides = [maybe_convert_scalar_to_list(conv_op, i) for i in strides]
                self.return_skips = return_skips
                
                # store other data about encoder that the decoder may use
                self.conv_op = conv_op
                self.norm_op = norm_op
                self.norm_op_kwargs = norm_op_kwargs
                self.nonlin = nonlin
                self.nonlin_kwargs = nonlin_kwargs
                self.dropout_op = dropout_op
                self.dropout_op_kwargs = dropout_op_kwargs
                self.conv_bias = conv_bias
                self.kernel_sizes = kernel_sizes
                self.scale_factors = interpScaleFactors
                self.n_conv_per_stage = n_conv_per_stage
                
    def forward(self, x):
        outputs = []
        # iterate through layers -- go one by one to get output for skip connections
        print(len(self.stages))
        for stage in self.stages:
            x = stage(x)
            outputs.append(x)
        if self.return_skips: # check if we want to return skip connections
            return outputs
        else:
            return outputs[-1] # if not, only return the output from the final stage
    
    # similar to the estimate in plain_conv_encoder, but accounts for our upsamples
    def compute_conv_feature_map_size(self, input_size):
        output = np.int64(0)
        for s in range(len(self.stages)):
            if isinstance(self.stages[s], nn.Sequential):
                for sq in self.stages[s]:
                    if hasattr(sq, 'compute_conv_feature_map_size'):
                        output += self.stages[s][-1].compute_conv_feature_map_size(input_size)
            else:
                output += self.stages[s].compute_conv_feature_map_size(input_size)
            
            input_size = [(i*k) // j for i, j, k in zip(input_size, self.strides[s], [self.scale_factors[s]]*len(self.strides[s]) )]
        return output
                
                
# class for interpolation -- since pytorch doesn't allow us to use it on it's own in a nn.sequential
class interpolate(torch.nn.Module):
    def __init__(self, scaleFactor, interpMode):
        super(interpolate, self).__init__()
        self.interp = torch.nn.functional.interpolate
        self.scaleFactor = scaleFactor
        self.interpMode = interpMode

    def forward(self, x):
        x = self.interp(x, scale_factor=self.scaleFactor, mode=self.interpMode)
        return x             
                
    def compute_conv_feature_map_size(self, input_size):
        # the interpolation to upsample is not a learnable parameter, so we just return 0
        return 0
