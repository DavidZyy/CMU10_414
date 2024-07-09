from typing import Optional
from ..autograd import NDArray
from ..autograd import Op, Tensor, Value, TensorOp
from ..autograd import TensorTuple, TensorTupleOp

from .ops_mathematic import *

from needle import ops
import numpy as array_api

class LogSoftmax(TensorOp):
    def compute(self, Z):
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION


def logsoftmax(a):
    return LogSoftmax()(a)


class LogSumExp(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, Z):
        # it seems that compute return a ndarray, so should only call array_api method.
        max_Z = array_api.max(Z, axis=self.axes, keepdims=True)
        shifted_Z = Z - max_Z
        exp_shifted_Z = array_api.exp(shifted_Z)
        sum_exp_shifted_Z = array_api.sum(exp_shifted_Z, axis=self.axes, keepdims=True)
        log_sum_exp_shifted_Z = array_api.log(sum_exp_shifted_Z) + max_Z
        squeeze_log_sum_exp_shifted_Z = np.squeeze(log_sum_exp_shifted_Z, axis=self.axes)
        return squeeze_log_sum_exp_shifted_Z

    # for the given example backward2
    # (3, 3, 3) is squeezed along axes 1,2 to (3,),
    # and recover its squeezed axes to (3, 1, 1)
    # and then broadcast it to (3, 3, 3)
    def gradient(self, out_grad, node):
        # gradient return a tensor, so could call ndl method
        # raise NotImplementedError()
        Z = node.inputs[0].cached_data
        max_Z = array_api.max(Z, axis=self.axes, keepdims=True)
        shifted_Z = Z - max_Z
        exp_shifted_Z = array_api.exp(shifted_Z)
        sum_exp_shifted_Z = array_api.sum(exp_shifted_Z, axis=self.axes, keepdims=True)
        softmax_Z = exp_shifted_Z / sum_exp_shifted_Z

        # first reshape, make sure have same dimensions
        # find the squeezed axes, and then recover it.
        reshape_shape = list(softmax_Z.shape)
        if self.axes is not None:
            for i in range(len(self.axes)):
                reshape_shape[self.axes[i]] = 1
        else:
            for i in range(len(reshape_shape)):
                reshape_shape[i] = 1
        reshape_shape = tuple(reshape_shape)
        # reshape the shape
        out_grad = out_grad.reshape(reshape_shape)
        # then broadcast, make sure the number on each dimension equal
        out_grad = out_grad.broadcast_to(softmax_Z.shape)

        return out_grad * Tensor(softmax_Z)


def logsumexp(a, axes=None):
    return LogSumExp(axes=axes)(a)

