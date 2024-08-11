from typing import Optional
from ..autograd import NDArray
from ..autograd import Op, Tensor, Value, TensorOp
from ..autograd import TensorTuple, TensorTupleOp

from .ops_mathematic import *

from ..backend_selection import array_api, BACKEND 

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
        ### BEGIN YOUR SOLUTION
        # max_Z = array_api.max(Z, axis=self.axes, keepdims=True)  # error, max is an inner method of array.
        input_shape = Z.shape
        if self.axes is None:
            reshape_shape = [1] * len(input_shape)
        elif isinstance(self.axes, int):
            reshape_shape = list(input_shape)
            reshape_shape[self.axes] = 1
            reshape_shape = tuple(reshape_shape)
        else:  # is tuple
            reshape_shape = list(input_shape)
            for i in self.axes:
                reshape_shape[i] = 1
            reshape_shape = tuple(reshape_shape)

        # for example, Z is (5, 3), axis = 0
        max_Z = Z.max(axis=self.axes, keepdims=False)  # (3, )
        max_Z_broadcast = max_Z.reshape(reshape_shape).broadcast_to(input_shape)  # (5, 3)
        shifted_Z = Z - max_Z_broadcast
        exp_shifted_Z = array_api.exp(shifted_Z)
        sum_exp_shifted_Z = array_api.sum(exp_shifted_Z, axis=self.axes, keepdims=False)  # (3, )
        log_sum_exp_shifted_Z = array_api.log(sum_exp_shifted_Z) + max_Z  # (3, )
        return log_sum_exp_shifted_Z
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        Z = node.inputs[0].cached_data
        max_Z = array_api.max(Z, axis=self.axes, keepdims=True)
        max_Z = max_Z.broadcast_to(Z.shape)  # NDArray do not support implicit broadcast, must broadcast explicit
        shifted_Z = Z - max_Z
        exp_shifted_Z = array_api.exp(shifted_Z)
        sum_exp_shifted_Z = array_api.sum(exp_shifted_Z, axis=self.axes, keepdims=True)
        sum_exp_shifted_Z = sum_exp_shifted_Z.broadcast_to(exp_shifted_Z.shape)
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

        # if not set device of Tensor, it will set to default_device, i.e. cpu_numpy().
        return out_grad * Tensor(softmax_Z, device=node.inputs[0].device, dtype=node.inputs[0].dtype)
        ### END YOUR SOLUTION


def logsumexp(a, axes=None):
    return LogSumExp(axes=axes)(a)

