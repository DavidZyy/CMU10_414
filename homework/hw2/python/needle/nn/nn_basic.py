"""The module.
"""
from typing import List, Callable, Any
from needle.autograd import Tensor
from needle import ops
import needle.init as init
import numpy as np


class Parameter(Tensor):
    """A special kind of tensor that represents parameters."""


def _unpack_params(value: object) -> List[Tensor]:
    if isinstance(value, Parameter):
        return [value]
    elif isinstance(value, Module):
        return value.parameters()
    elif isinstance(value, dict):
        params = []
        for k, v in value.items():
            params += _unpack_params(v)
        return params
    elif isinstance(value, (list, tuple)):
        params = []
        for v in value:
            params += _unpack_params(v)
        return params
    else:
        return []


def _child_modules(value: object) -> List["Module"]:
    if isinstance(value, Module):
        modules = [value]
        modules.extend(_child_modules(value.__dict__))
        return modules
    if isinstance(value, dict):
        modules = []
        for k, v in value.items():
            modules += _child_modules(v)
        return modules
    elif isinstance(value, (list, tuple)):
        modules = []
        for v in value:
            modules += _child_modules(v)
        return modules
    else:
        return []


class Module:
    def __init__(self):
        self.training = True

    def parameters(self) -> List[Tensor]:
        """Return the list of parameters in the module."""
        return _unpack_params(self.__dict__)

    def _children(self) -> List["Module"]:
        return _child_modules(self.__dict__)

    def eval(self):
        self.training = False
        for m in self._children():
            m.training = False

    def train(self):
        self.training = True
        for m in self._children():
            m.training = True

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class Identity(Module):
    def forward(self, x):
        return x


class Linear(Module):
    def __init__(
        self, in_features, out_features, bias=True, device=None, dtype="float32"
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        ### BEGIN YOUR SOLUTION
        # raise NotImplementedError()
        self.weight = init.kaiming_uniform(in_features, out_features, device=device, dtype=dtype, requires_grad=True)
        # bias should be one dimension
        if bias:
            # no need to pass in 1 for fan_out, or I need a transpose
            self.bias = init.kaiming_uniform(1,  out_features, device=device, dtype=dtype, requires_grad=True)
        else:
            self.bias = None
        ### END YOUR SOLUTION

    # X (N, H_in)
    # W (H_in, H_out)
    # bias (H_out)
    def forward(self, X: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        # raise NotImplementedError()
        if self.bias is not None:
            # for debug convenience
            '''
            maybe should check the shape of X and weight here! do broadcast_to 
            explicitly, not do it implicitly in numpy.
            '''
            temp1 = ops.matmul(X, self.weight)
            temp2 = ops.broadcast_to(self.bias, temp1.shape)
            temp3 = ops.add(temp1, temp2)
            result = temp3
        else:
            temp1 = ops.matmul(X, self.weight)
            result = temp1
        return result
        ### END YOUR SOLUTION


class Flatten(Module):
    def forward(self, X):
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION


class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        temp1 = ops.relu(x)
        result = temp1
        return result
        ### END YOUR SOLUTION


class Sequential(Module):
    def __init__(self, *modules):
        super().__init__()
        self.modules = modules

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        for module in self.modules:
            x = module(x)
        return x
        ### END YOUR SOLUTION


# how to get softmaxloss from LogSumExp?
# see: https://blog.csdn.net/yjw123456/article/details/121869249
class SoftmaxLoss(Module):
    def forward(self, logits: Tensor, y: Tensor):
        ### BEGIN YOUR SOLUTION
        # explicitly call broadcast
        temp1 = ops.logsumexp(logits, axes=(1,))
        a = list(temp1.shape)
        a.append(1)
        a = tuple(a)
        temp2 = ops.reshape(temp1, a)
        temp3 = ops.broadcast_to(temp2, logits.shape)
        temp4 = logits - temp3
        temp5 = ops.exp(temp4)  # softmax: exp_Z / exp_sum

        y_one_hot = init.one_hot(logits.shape[1], y)
        temp6 = temp5 * y_one_hot
        temp7 = ops.summation(temp6, axes=(1,))
        temp8 = -ops.log(temp7)
        temp9 = ops.summation(temp8) / logits.shape[0]

        return temp9
        # raise NotImplementedError()
        ### END YOUR SOLUTION


class BatchNorm1d(Module):
    def __init__(self, dim, eps=1e-5, momentum=0.1, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.momentum = momentum
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION


class LayerNorm1d(Module):
    def __init__(self, dim, eps=1e-5, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        ### BEGIN YOUR SOLUTION
        # raise NotImplementedError()
        self.weight = init.ones(dim, device=device, dtype=dtype, requires_grad=True)
        self.bias = init.zeros(dim, device=device, dtype=dtype, requires_grad=True)
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        # suppose x have shape of (n, k)
        n = x.shape[0]
        k = x.shape[1]
        temp1 = ops.summation(x, axes=(1,))  # (n, )
        temp2 = ops.reshape(temp1, (n, 1))  # (n, 1)
        temp3 = ops.broadcast_to(temp2, x.shape)  # (n, k)
        temp4 = temp3 / k  # E(x)

        temp5 = x - temp4
        temp6 = ops.power_scalar(temp5, 2)
        temp7 = ops.summation(temp6, axes=(1,))
        temp8 = ops.reshape(temp7, (n, 1))  # (n, 1)
        temp9 = ops.broadcast_to(temp8, x.shape) / k  # (n, k) Var(x)

        temp10 = temp9 + self.eps
        temp11 = ops.power_scalar(temp10, 0.5)

        temp12 = ops.divide(temp5, temp11)

        temp13 = ops.broadcast_to(self.weight, x.shape)
        temp14 = ops.broadcast_to(self.bias, x.shape)

        temp15 = temp13 * temp12 + temp14
        result = temp15
        return result

        # raise NotImplementedError()
        ### END YOUR SOLUTION


class Dropout(Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION


class Residual(Module):
    def __init__(self, fn: Module):
        super().__init__()
        self.fn = fn

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION
