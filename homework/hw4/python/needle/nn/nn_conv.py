"""The module.
"""
from typing import List, Callable, Any
from needle.autograd import Tensor
from needle import ops
import needle.init as init
import numpy as np
from .nn_basic import Parameter, Module


class Conv(Module):
    """
    Multi-channel 2D convolutional layer
    IMPORTANT: Accepts inputs in NCHW format, outputs also in NCHW format
    Only supports padding=same
    No grouped convolution or dilation
    Only supports square kernels
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, bias=True, device=None, dtype="float32"):
        super().__init__()
        if isinstance(kernel_size, tuple):
            kernel_size = kernel_size[0]
        if isinstance(stride, tuple):
            stride = stride[0]
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride

        ### BEGIN YOUR SOLUTION
        # raise NotImplementedError()
        kernel_shape = (self.kernel_size, self.kernel_size, self.in_channels, self.out_channels)
        kernel_array = init.kaiming_uniform(self.in_channels*self.kernel_size*self.kernel_size, 
                                            self.out_channels*self.kernel_size*self.kernel_size, 
                                            shape=kernel_shape)

        self.weight = Parameter(kernel_array, device=device, dtype=dtype)

        if bias:
            bound = 1.0 / (in_channels * kernel_size**2)**0.5
            bias_array = init.rand(self.out_channels, low=-bound, high=bound, device=device, dtype=dtype)
            self.bias = Parameter(bias_array, device=device, dtype=dtype)
        else:
            self.bias = None

        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        '''
        x is in NCHW format, should transpose it to NHWC format.
        how to calculate the padding (when stride = 1) ?
        (H + 2*pad) - k + 1 = H
        pad = (k - 1) / 2
        '''
        ### BEGIN YOUR SOLUTION
        x = x.transpose((1, 2)).transpose((2, 3))
        out = ops.conv(x, self.weight, self.stride, padding=self.kernel_size//2)
        N, H, W, C = out.shape
        assert C == self.out_channels
        if self.bias is not None:
            broadcast_bias = self.bias.reshape((1, 1, 1, self.out_channels)).broadcast_to((1, 1, W, self.out_channels)).broadcast_to((1, H, W, self.out_channels)).broadcast_to((N, H, W, self.out_channels))
            out = out + broadcast_bias
        out = out.transpose((2, 3)).transpose((1, 2))
        return out
        # raise NotImplementedError()
        ### END YOUR SOLUTION
