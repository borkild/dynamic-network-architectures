import torch
from typing import Type
import os
import torch.onnx as onnx

# annoying and dumb initial path adding if on windows
# import sys and add path with custom modules to search path
import sys
currentFilePath = os.getcwd()
if "win" in sys.platform: # hard code path, as it only lives on 1 windows machine right now
    pathToAdd = "C:\\Users\\Ben Orkild\\Documents\\CodeRepos\\dynamic-network-architectures"
    sys.path.append(pathToAdd)

from dynamic_network_architectures.architectures.abstract_arch import (
    AbstractDynamicNetworkArchitectures,
    test_submodules_loadable,
)

from dynamic_network_architectures.initialization.weight_init import InitWeights_He
from dynamic_network_architectures.building_blocks.kiNet_encoder import PlainUpsampleEncoder
from dynamic_network_architectures.building_blocks.kiNet_decoder import kiNetDecoder
from dynamic_network_architectures.building_blocks.helper import convert_conv_op_to_dim, get_matching_interp_mode

from torch import nn
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.dropout import _DropoutNd



class PlainConvKiNet(AbstractDynamicNetworkArchitectures):
    def __init__(self,
                input_channels: int,
                n_stages: int,
                features_per_stage: int | list[int] | tuple[int, ...],
                conv_op: Type[_ConvNd],
                kernel_sizes: int | list[int] | tuple[int, ...],
                strides: int | list[int] | tuple[int, ...],
                n_conv_per_stage: int | list[int] | tuple[int, ...],
                interp_scale_factors: int | list[int] | tuple[int, ...],
                num_classes: int,
                n_conv_per_stage_decoder: int | tuple[int, ...] | list[int],
                conv_bias: bool = False,
                norm_op: None | Type[nn.Module] = None,
                norm_op_kwargs: dict = None,
                dropout_op: None | Type[_DropoutNd] = None,
                dropout_op_kwargs: dict = None,
                nonlin: None | Type[torch.nn.Module] = None,
                nonlin_kwargs: dict = None,
                deep_supervision: bool = False,
                nonlin_first: bool = False ):
        super().__init__()
        
        self.key_to_encoder = "encoder.stages"  # Contains the stem as well.
        self.key_to_stem = "encoder.stages.0"
        self.keys_to_in_proj = (
            "encoder.stages.0.0.convs.0.all_modules.0",
            "encoder.stages.0.0.convs.0.conv",  # duplicate of above
        )
        # check for lists vs ints
        if isinstance(n_conv_per_stage, int):
            n_conv_per_stage = [n_conv_per_stage] * n_stages
        if isinstance(n_conv_per_stage_decoder, int):
            n_conv_per_stage_decoder = [n_conv_per_stage_decoder] * (n_stages - 1)
        assert len(n_conv_per_stage) == n_stages, (
            "n_conv_per_stage must have as many entries as we have "
            f"resolution stages. here: {n_stages}. "
            f"n_conv_per_stage: {n_conv_per_stage}"
        )
        assert len(n_conv_per_stage_decoder) == (n_stages - 1), (
            "n_conv_per_stage_decoder must have one less entries "
            f"as we have resolution stages. here: {n_stages} "
            f"stages, so it should have {n_stages - 1} entries. "
            f"n_conv_per_stage_decoder: {n_conv_per_stage_decoder}"
        )
        # get interpolation mode based on convolution type
        interp_mode = get_matching_interp_mode(conv_op)
        
        # create encoder
        self.encoder = PlainUpsampleEncoder(input_channels,
                                            n_stages,
                                            features_per_stage,
                                            conv_op,
                                            kernel_sizes,
                                            strides,
                                            n_conv_per_stage,
                                            interp_scale_factors,
                                            interp_mode,
                                            conv_bias,
                                            norm_op,
                                            norm_op_kwargs,
                                            dropout_op,
                                            dropout_op_kwargs,
                                            nonlin,
                                            nonlin_kwargs,
                                            nonlin_first,
                                            return_skips=True)
        
        # create decoder
        self.decoder = kiNetDecoder(self.encoder,
                                    num_classes,
                                    n_conv_per_stage_decoder,
                                    conv_op,
                                    nonlin_first,
                                    norm_op,
                                    norm_op_kwargs,
                                    dropout_op,
                                    dropout_op_kwargs,
                                    nonlin,
                                    nonlin_kwargs,
                                    conv_bias)
        
        
    # forward pass function
    def forward(self, x):
        skips = self.encoder(x)
        return self.decoder(skips)
    
    
    def compute_conv_feature_map_size(self, input_size):
        assert len(input_size) == convert_conv_op_to_dim(self.encoder.conv_op), (
            "just give the image size without color/feature channels or "
            "batch channel. Do not give input_size=(b, c, x, y(, z)). "
            "Give input_size=(x, y(, z))!"
        )
        # sum encoder and decoder feature map sizes
        return self.encoder.compute_conv_feature_map_size(input_size) + self.decoder.compute_conv_feature_map_size(input_size)
    
    
    @staticmethod
    def initialize(module):
        InitWeights_He(1e-2)(module)
        
    
    
    
# test architecture
if __name__ == "__main__":
    tstData = torch.rand((1, 2, 220, 220), requires_grad=True)
    
    net = PlainConvKiNet(2,
                         3,
                         [32, 64, 128],
                         nn.Conv2d,
                         3,
                         1,
                         2,
                         2,
                         2,
                         2,
                         conv_bias=True,
                         norm_op=nn.BatchNorm2d,
                         norm_op_kwargs=None,
                         nonlin=nn.ReLU)
    
    # test if submodules can load
    test_submodules_loadable(net)
    # test feature map size calculation
    print(net.compute_conv_feature_map_size(tstData.shape[2:]))
    # test forward pass
    tstOut = net(tstData)
    
    # save example model out
    saveModel = True
    savePath = "C:\\Users\\Ben Orkild\\Downloads\\modelweights.onnx"
    if saveModel:
        torch.onnx.export(net,
                          tstData,
                          savePath,
                          export_params=True)
    



