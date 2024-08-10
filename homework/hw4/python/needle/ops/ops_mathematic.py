"""Operator implementations."""

from numbers import Number
from typing import Optional, List, Tuple, Union

from ..autograd import NDArray
from ..autograd import Op, Tensor, Value, TensorOp
from ..autograd import TensorTuple, TensorTupleOp
import numpy

# NOTE: we will import numpy as the array_api
# as the backend for our computations, this line will change in later homeworks

# import numpy as array_api # not correct!!
from ..backend_selection import array_api, BACKEND
from .ops_tuple import *


class EWiseAdd(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return (a + b)
        # return (a + b).astype(a.dtype)

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad, out_grad


def add(a, b):
    return EWiseAdd()(a, b)


class AddScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return (a + self.scalar)
        # return (a + self.scalar).astype(a.dtype)

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad


def add_scalar(a, scalar):
    return AddScalar(scalar)(a)


class EWiseMul(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a * b

    def gradient(self, out_grad: Tensor, node: Tensor):
        lhs, rhs = node.inputs
        return out_grad * rhs, out_grad * lhs


def multiply(a, b):
    return EWiseMul()(a, b)


class MulScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return (a * self.scalar)
        # return (a * self.scalar).astype(a.dtype)

    def gradient(self, out_grad: Tensor, node: Tensor):
        return (out_grad * self.scalar,)


def mul_scalar(a, scalar):
    return MulScalar(scalar)(a)


class PowerScalar(TensorOp):
    """Op raise a tensor to an (integer) power."""

    def __init__(self, scalar: int):
        self.scalar = scalar

    def compute(self, a: NDArray) -> NDArray:
        return (a ** self.scalar)
        # return (a ** self.scalar).astype(a.dtype)

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        # raise NotImplementedError()
        return (out_grad * self.scalar * (node.inputs[0] ** (self.scalar - 1)),)
        ### END YOUR SOLUTION


def power_scalar(a, scalar):
    return PowerScalar(scalar)(a)


class EWisePow(TensorOp):
    """Op to element-wise raise a tensor to a power."""

    def compute(self, a: NDArray, b: NDArray) -> NDArray:
        return a ** b

    def gradient(self, out_grad, node):
        if not isinstance(node.inputs[0], NDArray) or not isinstance(
                node.inputs[1], NDArray
        ):
            raise ValueError("Both inputs must be tensors (NDArray).")

        a, b = node.inputs[0], node.inputs[1]
        grad_a = out_grad * b * (a ** (b - 1))
        grad_b = out_grad * (a ** b) * array_api.log(a.data)
        return grad_a, grad_b


def power(a, b):
    return EWisePow()(a, b)


class EWiseDiv(TensorOp):
    """Op to element-wise divide two nodes."""

    def compute(self, a, b):
        return a / b

    def gradient(self, out_grad, node):
        X, Y = node.inputs[0], node.inputs[1]
        grad_X = out_grad * 1 / Y
        grad_Y = out_grad * -X / Y ** 2
        return grad_X, grad_Y


def divide(a, b):
    return EWiseDiv()(a, b)


class DivScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a):
        return (a / self.scalar)
        # return (a / self.scalar).astype(a.dtype)

    def gradient(self, out_grad, node):
        return (out_grad * 1 / self.scalar,)


def divide_scalar(a, scalar):
    return DivScalar(scalar)(a)


# Transpose: reverses the order of two axes (axis1, axis2), defaults to the last two axes (1 input, axes - tuple)
class Transpose(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        new_axes = [i for i in range(len(a.shape))]
        if self.axes is None:
            new_axes[-1], new_axes[-2] = new_axes[-2], new_axes[-1]
        else:
            new_axes[self.axes[0]], new_axes[self.axes[1]] = new_axes[self.axes[1]], new_axes[self.axes[0]]
        result = a.permute(tuple(new_axes))
        return result

    def gradient(self, out_grad, node):
        # return Tensor(array_api.swapaxes(out_grad.cached_data, self.axes[0], self.axes[1]))
        return transpose(out_grad, self.axes)


def transpose(a, axes=None):
    return Transpose(axes)(a)


class Reshape(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        return array_api.reshape(a, self.shape)

    def gradient(self, out_grad, node: Tensor):
        return reshape(out_grad, node.inputs[0].shape)


def reshape(a, shape):
    return Reshape(shape)(a)


class BroadcastTo(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        return array_api.broadcast_to(a, self.shape)

    def gradient(self, out_grad, node):
        # the len of out_grad.shape(self.shape) usually >= node.inputs[0].shape
        # sum along the axis that not equal
        shape1, shape2 = self.shape, node.inputs[0].shape
        len1 = len(shape1)
        len2 = len(shape2)
        shape2_prepend = [1] * (len1 - len2) + list(
            shape2)  # prepend 1s to shape2 if len1 != len2, refer the broadcasting rule
        axis = []
        for i in range(len1):
            if shape1[i] != shape2_prepend[i]:
                axis.append(i)
        temp1 = summation(out_grad, axes=tuple(axis))
        temp2 = reshape(temp1, node.inputs[0].shape)
        return temp2
        # return reshape(summation(out_grad, axes=tuple(axis)), node.inputs[0].shape)


def broadcast_to(a, shape):
    return BroadcastTo(shape)(a)


class Summation(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    # NOTE the summation method not keep dimension, so if we need to recover Tensor's original shape
    # when backward, should reshape and then broadcast, for example:
    # (3, 3) --summation(axes = (1,)) --> (3,)
    # (3,) --reshape--> (3,1) --broadcast--> (3,3)
    def compute(self, a):
        return array_api.sum(a, self.axes, keepdims=False)

    # def gradient(self, out_grad, node):
    #     input_shape = node.inputs[0].shape
    #     output_shape = out_grad.shape
    #     reshape_shape = []
    #     for i in input_shape:
    #         if i in output_shape:
    #             reshape_shape.append(i)
    #         else:
    #             reshape_shape.append(1)
    #     return broadcast_to(reshape(out_grad, reshape_shape), input_shape)
    def gradient(self, out_grad, node):
        input_shape = node.inputs[0].shape
        input_shape_list = list(input_shape)

        if self.axes is None:
            for i in range(len(input_shape)):
                input_shape_list[i] = 1
        elif isinstance(self.axes, int):
            input_shape_list[self.axes] = 1
        else:  # is tuple
            for i in self.axes:
                input_shape_list[i] = 1

        reshape_shape = tuple(input_shape_list)
        temp1 = reshape(out_grad, reshape_shape)
        result = broadcast_to(temp1, input_shape)

        return result


def summation(a, axes=None):
    return Summation(axes)(a)


class MatMul(TensorOp):
    def compute(self, a, b):
        # return array_api.matmul(a, b)
        return a @ b

    def gradient(self, out_grad, node):
        mat_a, mat_b = node.inputs
        if len(mat_a.shape) > len(mat_b.shape):
            axes = tuple(range(len(mat_a.shape) - len(mat_b.shape)))
            return matmul(out_grad, mat_b.transpose()), summation(matmul(mat_a.transpose(), out_grad), axes=axes)
        elif len(mat_a.shape) < len(mat_b.shape):
            axes = tuple(range(len(mat_b.shape) - len(mat_a.shape)))
            return summation(matmul(out_grad, mat_b.transpose()), axes=axes), matmul(mat_a.transpose(), out_grad)
        else:
            return matmul(out_grad, mat_b.transpose()), matmul(mat_a.transpose(), out_grad)


def matmul(a, b):
    return MatMul()(a, b)


class Negate(TensorOp):
    def compute(self, a):
        return -a

    def gradient(self, out_grad, node):
        return -out_grad


def negate(a):
    return Negate()(a)


class Log(TensorOp):
    def compute(self, a):
        return array_api.log(a)

    def gradient(self, out_grad, node):
        return out_grad * 1 / node.inputs[0]


def log(a):
    return Log()(a)


class Exp(TensorOp):
    def compute(self, a):
        return array_api.exp(a)

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad * exp(node.inputs[0])


def exp(a):
    return Exp()(a)


class ReLU(TensorOp):
    def compute(self, a):
        return array_api.maximum(a, 0)

    def gradient(self, out_grad, node):
        return Tensor(node.inputs[0].numpy() > 0, dtype='float32') * out_grad


def relu(a):
    return ReLU()(a)


class Tanh(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        # raise NotImplementedError()
        return array_api.tanh(a)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        temp1 = tanh(node.inputs[0]) ** 2
        # temp2 = 1 - temp1  # get bug here !!! scalar have no method of sub Tensor
        temp2 = - temp1 + 1  # right!
        result = out_grad * temp2
        return result

        # temp1 = reshape(
        #         Tensor((1,), dtype="float32", device=node.device),
        #         len(node.inputs[0].shape) * (1,)
        # )
        # temp2 = broadcast_to(temp1, node.shape)
        # temp3 = temp2 - tanh(node.inputs[0]) ** 2
        # result = temp3 * out_grad
        # return result

        ## END YOUR SOLUTION


def tanh(a):
    return Tanh()(a)


class Stack(TensorOp):
    def __init__(self, axis: int):
        """
        Concatenates a sequence of arrays along a new dimension.
        Parameters:
        axis - dimension to concatenate along
        All arrays need to be of the same size.
        """
        self.axis = axis

    # stack tensor A of shape (5, 5) and tensor B of shape (5, 5) 
    # on axis = 1 get tensor C of shape (5, 2, 5) for example.
    # we need to execute the following steps:
    # C = zeros(5, 2, 5)
    # C[:, 0, :] = A
    # C[:, 1, :] = B
    # def compute(self, args: tuple) -> ndarray:
    def compute(self, args: TensorTuple) -> Tensor:
        ### BEGIN YOUR SOLUTION
        # raise NotImplementedError()
        # out = NDArray.make(self.shape, device=self.device)
        # return array_api.stack(args, self.axis)
        # make ndarray
        shape = list(args[0].shape)  # the shape of tensor to be stacked :(5, 5)
        shape = shape[:self.axis] + [len(args)] + shape[self.axis:]  # the new shape after stack :(5, 2, 5)
        result = array_api.empty(shape, device=args[0].device)  # C = zeros(5, 2, 5)

        # get [slice(0, 5, 1), 0, slice(0, 5, 1)], also is [:, 0, :]
        idx = []
        for i in range(len(shape)):
            if i == self.axis:
                idx.append(0) # 0 act as a placeholder
            else:
                idx.append(slice(0, shape[i], 1))

        for i in range(len(args)):
            idx[self.axis] = i  # get [:, 0, :] and [:, 1, :]
            result[tuple(idx)] = args[i]  # C[:, 0, :] = A, C[:, 1, 0] = B
        return result
        ### END YOUR SOLUTION


    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        # raise NotImplementedError()
        return split(out_grad, self.axis)
        # temp = split(out_grad, self.axis)
        # length = len(node.inputs[0])
        # return make_tuple(*(temp[i] for i in range(length)))
        ### END YOUR SOLUTION


def stack(args, axis):
    # return Stack(axis)(make_tuple(*args))
    temp1 = make_tuple(*args)
    return Stack(axis)(temp1)


class Split(TensorTupleOp):
    def __init__(self, axis: int):
        """
        Splits a tensor along an axis into a tuple of tensors.
        (The "inverse" of Stack)
        Parameters:
        axis - dimension to split
        """
        self.axis = axis

    # for example (5, 2, 5) -> (5, 5) and (5, 5)
    def compute(self, A):
        ### BEGIN YOUR SOLUTION
        # the inverse of stack
        shape = list(A.shape)  # (5, 2, 5)

        idx = []
        for i in range(len(shape)):
            if i == self.axis:
                idx.append(0)  # 0 act as a placeholder
            else:
                idx.append(slice(0, shape[i], 1))

        num = shape.pop(self.axis)  # shape become (5, 5), shape is changed here!

        result = []
        for i in range(num):
            idx[self.axis] = i  # get [:, 0, :] and [:, 1, :]
            # A = C[:, 0, :], B = C[:, 1, :], the shape is (5, 1, 5), should reshape it to (5, 5)
            a = A[tuple(idx)].compact()  # if you have no compact, the reshape will get error
            result.append(a.reshape(shape))

        return tuple(result)  # return (A, B)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return stack(out_grad, axis=self.axis)
        ### END YOUR SOLUTION


def split(a, axis):
    return Split(axis)(a)


class Flip(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    # return a ndarray
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.flip(a, self.axes)
        ### END YOUR SOLUTION

    # return a Tensor
    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        # raise NotImplementedError()
        result = flip(out_grad, self.axes)
        return result
        ### END YOUR SOLUTION


def flip(a, axes):
    return Flip(axes)(a)


class Dilate(TensorOp):
    def __init__(self, axes: tuple, dilation: int):
        self.axes = axes
        self.dilation = dilation

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        shape = []
        index = []
        for i in range(len(a.shape)):
            if i in self.axes:
                shape.append(a.shape[i] * (1 + self.dilation))
                index.append(slice(0, a.shape[i] * (1 + self.dilation), (1 + self.dilation)))
            else:
                shape.append(a.shape[i])
                index.append(slice(0, a.shape[i], 1))

        result = array_api.empty(shape=shape, device=a.device)
        result.fill(0)
        result[tuple(index)] = a
        return result
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        # raise NotImplementedError()
        return undilate(out_grad, self.axes, self.dilation)
        ### END YOUR SOLUTION


def dilate(a, axes, dilation):
    return Dilate(axes, dilation)(a)


class UnDilate(TensorOp):
    def __init__(self, axes: tuple, dilation: int):
        self.axes = axes
        self.dilation = dilation

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        # raise NotImplementedError()
        shape = []
        index = []
        for i in range(len(a.shape)):
            if i in self.axes:
                shape.append(a.shape[i] // (1 + self.dilation))
                index.append(slice(0, a.shape[i], (1 + self.dilation)))
            else:
                shape.append(a.shape[i])
                index.append(slice(0, a.shape[i], 1))
        
        result = array_api.empty(shape=shape, device=a.device)
        result.fill(0)
        result = a[tuple(index)]
        return result
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        # raise NotImplementedError()
        return dilate(out_grad, self.axes, self.dilation)
        ### END YOUR SOLUTION


def undilate(a, axes, dilation):
    return UnDilate(axes, dilation)(a)


class Conv(TensorOp):
    def __init__(self, stride: Optional[int] = 1, padding: Optional[int] = 0):
        self.stride = stride
        self.padding = padding

    def compute(self, A, B):
        ### BEGIN YOUR SOLUTION
        '''
        method 1: expand(tile) A (X)
        '''
        # pad A first
        # pad_A = A.pad(((0, 0), (self.padding, self.padding), (self.padding, self.padding), (0, 0)))
        #
        # N, H, W, C_in = pad_A.shape
        # K, _, _, C_out = B.shape
        # Ns, Hs, Ws, Cs = pad_A.strides
        #
        # inner_dim = K * K * C_in
        # new_H = (H - K) // self.stride + 1
        # new_W = (W - K) // self.stride + 1
        #
        # new_shape = (N, new_H, new_W, K, K, C_in)
        # new_strides = (Ns, Hs * self.stride, Ws * self.stride, Hs, Ws, Cs)
        # temp1 = NDArray.make(shape=new_shape, strides=new_strides, device=A.device, handle=pad_A._handle)
        # temp2 = temp1.compact()  # use more memory
        # temp3 = temp2.reshape((N*(new_H)*(new_W), inner_dim))  # not support (-1, inner_dim)
        #
        # temp4 = B.reshape((inner_dim, C_out))
        # temp5 = temp3 @ temp4
        # temp6 = temp5.reshape((N, new_H, new_W, C_out))
        # return temp6

        '''
        method 2: expand(tile) B (X)
        '''
        pad_A = A.pad(((0, 0), (self.padding, self.padding), (self.padding, self.padding), (0, 0)))
        N, H, W, C_in = pad_A.shape
        K, _, _, C_out = B.shape

        new_H = (H - K) // self.stride + 1
        new_W = (W - K) // self.stride + 1

        reshape_pad_A = pad_A.reshape((N, H*W*C_in))

        # make an empty tensor with nothing for stacking
        B_list = []
        for i in range(new_H):
            for j in range(new_W):
                pad_B = B.pad(((i*self.stride, H-i*self.stride-K), (j*self.stride, W-j*self.stride-K), (0, 0), (0, 0)))
                reshape_pad_B = pad_B.reshape((H*W*C_in, C_out))
                B_list.append(reshape_pad_B)

        stackOp = Stack(axis=1)
        # stack_reshape_pad_B = stackOp(B_list)
        # stack_reshape_pad_B = Stack(2)(B_list)
        stack_reshape_pad_B = stackOp.compute(B_list)

        reshape_stack_reshape_pad_B = stack_reshape_pad_B.reshape((H*W*C_in, new_H*new_W*C_out))
        temp0 = reshape_pad_A @ reshape_stack_reshape_pad_B
        temp1 = temp0.reshape((N, new_H, new_W, C_out))
        return temp1
        # raise NotImplementedError()
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
#         A = node.inputs[0]
#         B = node.inputs[1]
# 
#         pad_A = A.pad(((0, 0), (self.padding, self.padding), (self.padding, self.padding), (0, 0)))
#         N, H, W, C_in = pad_A.shape
#         K, _, _, C_out = B.shape
#         Ns, Hs, Ws, Cs = pad_A.strides
# 
#         inner_dim = K * K * C_in
#         new_H = (H - K) // self.stride + 1
#         new_W = (W - K) // self.stride + 1
# 
#         temp1 = out_grad.reshape((N*new_H*new_W, C_out))
#         temp2 = 1

        ### END YOUR SOLUTION


# functions like this receive Tensor(s) and also return Tensor.
# it will call compute to calculate the NDArray of the Tensor.
def conv(a, b, stride=1, padding=1):
    return Conv(stride, padding)(a, b)
