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
        self.weight = Parameter(init.kaiming_uniform(in_features, out_features, device=device, dtype=dtype, requires_grad=True))
        # bias should be one dimension
        if bias:
            # no need to pass in 1 for fan_out, or I need a transpose
            self.bias = Parameter(init.kaiming_uniform(1,  out_features, device=device, dtype=dtype, requires_grad=True))
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
        # shape is a tuple
        old_shape = X.shape
        new_shape = (old_shape[0], np.prod(old_shape[1:]))
        result = ops.reshape(X, new_shape)
        return result
        # raise NotImplementedError()
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
        temp9 = ops.summation(temp8)
        temp10 = temp9 / logits.shape[0]

        return temp10
        # raise NotImplementedError()
        ### END YOUR SOLUTION


class BatchNorm1d(Module):
    def __init__(self, dim, eps=1e-5, momentum=0.1, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.momentum = momentum
        ### BEGIN YOUR SOLUTION
        self.weight = Parameter(init.ones(dim, device=device, dtype=dtype, requires_grad=True))
        self.bias = Parameter(init.zeros(dim, device=device, dtype=dtype, requires_grad=True))

        self.running_mean = init.zeros(dim, device=device, dtype=dtype, requires_grad=False)  # (dim, )
        self.running_var = init.ones(dim, device=device, dtype=dtype, requires_grad=False)  # (dim, )
        # raise NotImplementedError()
        ### END YOUR SOLUTION

    # not the calculation is along the batch dimension, which is different from LayerNorm1d
    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        n = x.shape[0]
        k = x.shape[1]
        if self.training:
            temp1 = ops.summation(x, axes=(0,))  # (k, )
            temp2 = ops.reshape(temp1, (1, k))  # (1, k)
            temp3 = ops.broadcast_to(temp2, x.shape)  # (n, k)
            batch_mean_n_k = temp3 / n
            batch_mean_k = temp1 / n

            temp4 = x - batch_mean_n_k
            temp5 = ops.power_scalar(temp4, 2)
            temp6 = ops.summation(temp5, axes=(0,))  # (k, )
            batch_var_k = temp6 / n

            # very interesting here, if not use .data method of batch mean and var here, the test memory of
            # adam optimizer will fail.
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * batch_mean_k.data
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * batch_var_k.data

            mean = batch_mean_k
            var = batch_var_k
        else:
            mean = self.running_mean
            var = self.running_var
        
        temp7 = ops.reshape(mean, (1, k))
        temp8 = ops.broadcast_to(temp7, x.shape)  # E(x)

        temp9 = ops.reshape(var, (1, k))
        temp10 = ops.broadcast_to(temp9, x.shape)  # var(x)

        temp11 = temp10 + self.eps
        temp12 = ops.power_scalar(temp11, 0.5)

        temp13 = x - temp8

        temp14 = ops.divide(temp13, temp12)

        temp15 = ops.broadcast_to(self.weight, x.shape)
        temp16 = ops.broadcast_to(self.bias, x.shape)

        result = temp15 * temp14 + temp16
        return result

        # raise NotImplementedError()
        ### END YOUR SOLUTION


class LayerNorm1d(Module):
    def __init__(self, dim, eps=1e-5, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        ### BEGIN YOUR SOLUTION
        # raise NotImplementedError()
        self.weight = Parameter(init.ones(dim, device=device, dtype=dtype, requires_grad=True))
        self.bias = Parameter(init.zeros(dim, device=device, dtype=dtype, requires_grad=True))
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
        if self.training:
            # Create a mask with the same shape as x, with elements 0 with probability p and 1 with probability (1 - p)
            mask = np.random.binomial(1, 1 - self.p, size=x.shape)
            mask_tensor = Tensor(mask)
            # Scale the output by 1 / (1 - p)
            temp1 = x * mask_tensor
            result = temp1 / (1 - self.p)
            return result
        else:
            return x
        # raise NotImplementedError()
        ### END YOUR SOLUTION


class Residual(Module):
    def __init__(self, fn: Module):
        super().__init__()
        self.fn = fn

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return x + self.fn(x)
        # raise NotImplementedError()
        ### END YOUR SOLUTION
