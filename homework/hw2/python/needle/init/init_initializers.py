import math
from .init_basic import *


def xavier_uniform(fan_in, fan_out, gain=1.0, **kwargs):
    ### BEGIN YOUR SOLUTION
    a = gain * math.sqrt(6 / (fan_in + fan_out))
    result = rand(fan_in, fan_out, low=-a, high=a, **kwargs)
    return result
    ### END YOUR SOLUTION


def xavier_normal(fan_in, fan_out, gain=1.0, **kwargs):
    ### BEGIN YOUR SOLUTION
    std = gain * math.sqrt(2 / (fan_in + fan_out))
    result = randn(fan_in, fan_out, mean=0, std=std, **kwargs)
    return result
    ### END YOUR SOLUTION


def kaiming_uniform(fan_in, fan_out, nonlinearity="relu", **kwargs):
    assert nonlinearity == "relu", "Only relu supported currently"
    ### BEGIN YOUR SOLUTION
    # divisor = fan_out if (fan_in == 1) else fan_in
    divisor = fan_out if (fan_in == 0) else fan_in
    gain = math.sqrt(2)
    bound = gain * math.sqrt(3 / divisor)
    result = rand(fan_in, fan_out, low=-bound, high=bound, **kwargs)
    return result
    ### END YOUR SOLUTION


def kaiming_normal(fan_in, fan_out, nonlinearity="relu", **kwargs):
    assert nonlinearity == "relu", "Only relu supported currently"
    ### BEGIN YOUR SOLUTION
    gain = math.sqrt(2)
    std = gain * math.sqrt(1 / fan_in)
    result = randn(fan_in, fan_out, mean=0, std=std, **kwargs)
    return result
    ### END YOUR SOLUTION
