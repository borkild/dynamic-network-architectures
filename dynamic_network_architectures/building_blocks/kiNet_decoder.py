import torch
from torch import nn
import numpy as np
from typing import Type

from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.dropout import _DropoutNd
from dynamic_network_architectures.building_blocks.simple_conv_blocks import StackedConvBlocks
from dynamic_network_architectures.building_blocks.kiNet_encoder import PlainUpsampleEncoder
from dynamic_network_architectures.building_blocks.helper import maybe_convert_scalar_to_list, get_matching_pool_op


class kiNetDecoder(torch.nn.Module):
    def __init__(self,
                 encoder: PlainUpsampleEncoder,
                 num_classes: int,
                 n_conv_per_stage: int | tuple[int, ...] | list[int],
                 conv_op: None | Type[_ConvNd] = None,
                 nonlin_first: bool = False,
                 norm_op: None | Type[nn.Module] = None,
                 norm_op_kwargs: dict = None,
                 dropout_op: None | Type[_DropoutNd] = None,
                 dropout_op_kwargs: dict = None,
                 nonlin: None | Type[nn.Module] = None,
                 nonlin_kwargs: dict = None,
                 conv_bias: bool = None ):
        
        super().__init__()
        self.encoder = encoder
        self.num_classes = num_classes
        n_stages_encoder = len(encoder.outputChannels)
        if isinstance(n_conv_per_stage, int):
            n_conv_per_stage = [n_conv_per_stage] * (n_stages_encoder - 1)
        assert len(n_conv_per_stage) == n_stages_encoder - 1, "n_conv_per_stage must have as many entries as we have " \
                                                          "resolution stages - 1 (n_stages in encoder - 1), " \
                                                          "here: %d" % n_stages_encoder
        
        # check if we gave other options to overwrite encoder given options
        conv_op = encoder.conv_op if conv_op is None else conv_op
        conv_bias = encoder.conv_bias if conv_bias is None else conv_bias
        norm_op = encoder.norm_op if norm_op is None else norm_op
        norm_op_kwargs = encoder.norm_op_kwargs if norm_op_kwargs is None else norm_op_kwargs
        dropout_op = encoder.dropout_op if dropout_op is None else dropout_op
        dropout_op_kwargs = encoder.dropout_op_kwargs if dropout_op_kwargs is None else dropout_op_kwargs
        nonlin = encoder.nonlin if nonlin is None else nonlin
        nonlin_kwargs = encoder.nonlin_kwargs if nonlin_kwargs is None else nonlin_kwargs
        
        # starts at largest size, and work our way down
        stages = []
        maxPools = []
        for curStage in range(0, n_stages_encoder):
            input_features_below = encoder.outputChannels[-(curStage+1)]
            if curStage + 2 > n_stages_encoder: # check if we're on our last layer of encoder, if so our next features is the numnber of classes
                input_features_above = num_classes
            else:
                input_features_above = encoder.outputChannels[-(curStage+2)]
            factorForPool = encoder.scale_factors[-curStage]
            # get max pooling operation for this stage
            maxPools.append( get_matching_pool_op(conv_op, pool_type="max")(kernel_size=factorForPool, stride=factorForPool) )
            # get convolutions for stage
            stages.append( StackedConvBlocks(n_conv_per_stage[curStage-1],
                                            encoder.conv_op, input_features_below,
                                            input_features_above,
                                            encoder.kernel_sizes[-(curStage + 1)], 
                                            encoder.strides[-curStage],
                                            conv_bias,
                                            norm_op,
                                            norm_op_kwargs,
                                            dropout_op,
                                            dropout_op_kwargs,
                                            nonlin,
                                            nonlin_kwargs,
                                            nonlin_first ) )
        
        self.stages = nn.ModuleList(stages)
        self.maxPools = nn.ModuleList(maxPools)
        
        
    def forward(self, skips):
        # take skip connections as input, should be in order they are computed in the encoder
        
        # last element in skips should be from the largest output
        curInput = skips[-1]
        # iterate through decoder stages
        for decodeStage in range(len(self.stages)):
            # apply convolutions first
            curInput = self.stages[decodeStage](curInput)
            # apply max pool next
            curInput =  self.maxPools[decodeStage](curInput)
            # don't add skips after last layer
            if decodeStage != len(self.stages) -1:
                # add max pooled input for skip connection now add skip connection before next stage
                curInput = torch.add(curInput, skips[-(decodeStage+2)])
        
        # return output
        return curInput
    
    
    def compute_conv_feature_map_size(self, input_size):
        # first we need to compute the skip sizes. Skip bottleneck because all output feature maps of our ops will at
        # least have the size of the skip above that (therefore -1)
        skip_sizes = []
        for s in range(len(self.encoder.strides) - 1):
            skip_sizes.append([(i*k) // j for i, j, k in zip(input_size, self.encoder.strides[s], [self.encoder.scale_factors[s]]*len(self.encoder.strides[s]) )])
            input_size = skip_sizes[-1]
        # print(skip_sizes)
        
        # since we're downsampling, we follow the plain_conv_encoder way of calculating input size and feature map size
        output = np.int64(0)
        for s in range(len(self.stages)):
            if isinstance(self.stages[s], nn.Sequential):
                for sq in self.stages[s]:
                    if hasattr(sq, 'compute_conv_feature_map_size'):
                        output += self.stages[s][-1].compute_conv_feature_map_size(input_size)
            else:
                output += self.stages[s].compute_conv_feature_map_size(input_size)
            input_size = [i // j for i, j in zip(input_size, [self.encoder.scale_factors[-(s+1)]]*len(input_size)  )]
        
        return output
        
        
        
            
            
            