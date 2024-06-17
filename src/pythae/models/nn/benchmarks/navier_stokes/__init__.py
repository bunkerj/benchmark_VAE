"""A collection of Neural nets used to perform the benchmark on NavierStokes"""

from .resnets import *

__all__ = [
    "Encoder_Conv_VAE_NavierStokes",
    "Decoder_ResNet_AE_NavierStokes",
]
