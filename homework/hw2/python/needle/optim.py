"""Optimization module"""
import needle as ndl
import numpy as np


class Optimizer:
    def __init__(self, params):
        self.params = params

    def step(self):
        raise NotImplementedError()

    def reset_grad(self):
        for p in self.params:
            p.grad = None


class SGD(Optimizer):
    def __init__(self, params, lr=0.01, momentum=0.0, weight_decay=0.0):
        super().__init__(params)
        self.lr = lr
        self.momentum = momentum
        self.u = {param: 0 for param in self.params}
        self.weight_decay = weight_decay

    def step(self):
        for param in self.params:
            if param.grad is None:
                continue

            if self.weight_decay != 0:
                grad = param.grad.data + self.weight_decay * param.data
            else:
                grad = param.grad.data

            self.u[param] = self.momentum * self.u[param] + (1 - self.momentum) * grad

            param.data -= self.lr * self.u[param]

    def clip_grad_norm(self, max_norm=0.25):
        """
        Clips gradient norm of parameters.
        """
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION


class Adam(Optimizer):
    def __init__(
        self,
        params,
        lr=0.01,
        beta1=0.9,
        beta2=0.999,
        eps=1e-8,
        weight_decay=0.0,
    ):
        super().__init__(params)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = 0

        # self.m = {}
        self.u = {param: 0 for param in self.params}
        self.v = {param: 0 for param in self.params}

    def step(self):
        self.t += 1
        for param in self.params:
            if param.grad is None:
                continue
            
            if self.weight_decay != 0:
                grad = param.grad.data + self.weight_decay * param.data
            else:
                grad = param.grad.data

            self.u[param] = self.beta1 * self.u[param] + (1 - self.beta1) * grad
            self.v[param] = self.beta2 * self.v[param] + (1 - self.beta2) * (grad ** 2)

            u_hat = self.u[param] / (1 - self.beta1 ** self.t)
            v_hat = self.v[param] / (1 - self.beta2 ** self.t)

            # param.data -= self.lr * u_hat / (v_hat ** 0.5 + self.eps)
            param.data = param.data - self.lr * u_hat / (v_hat ** 0.5 + self.eps)
