"""The module.
"""
from typing import List

import numpy

from needle.autograd import Tensor
from needle import ops
import needle.init as init
import numpy as np
from .nn_basic import Parameter, Module
from ..backend_selection import array_api, BACKEND

class Sigmoid(Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        # ops. is for tensor, array_api. is for NDArray
        temp1 = ops.exp(-x)
        temp2 = 1 + temp1

        # reuslt = 1 / temp2  # error
        one = init.ones(*x.shape, device=x.device, dtype=x.dtype)
        result = one / temp2
        return result
        ### END YOUR SOLUTION

def sigmoid(x):
    return Sigmoid()(x)

class RNNCell(Module):
    def __init__(self, input_size, hidden_size, bias=True, nonlinearity='tanh', device=None, dtype="float32"):
        """
        Applies an RNN cell with tanh or ReLU nonlinearity.

        Parameters:
        input_size: The number of expected features in the input X
        hidden_size: The number of features in the hidden state h
        bias: If False, then the layer does not use bias weights
        nonlinearity: The non-linearity to use. Can be either 'tanh' or 'relu'.

        Variables:
        W_ih: The learnable input-hidden weights of shape (input_size, hidden_size).
        W_hh: The learnable hidden-hidden weights of shape (hidden_size, hidden_size).
        bias_ih: The learnable input-hidden bias of shape (hidden_size,).
        bias_hh: The learnable hidden-hidden bias of shape (hidden_size,).

        Weights and biases are initialized from U(-sqrt(k), sqrt(k)) where k = 1/hidden_size
        """
        super().__init__()
        ### BEGIN YOUR SOLUTION
        k = 1 / hidden_size
        bound = (k)**0.5

        W_ih_array = init.rand(input_size, hidden_size, low=-bound, high=bound, device=device, dtype=dtype)
        self.W_ih = Parameter(W_ih_array, device=device, dtype=dtype)

        W_hh_array = init.rand(hidden_size, hidden_size, low=-bound, high=bound, device=device, dtype=dtype)
        self.W_hh = Parameter(W_hh_array, device=device, dtype=dtype)

        if bias:
            bias_ih_array = init.rand(hidden_size, device=device, dtype=dtype)
            self.bias_ih= Parameter(bias_ih_array, device=device, dtype=dtype)

            bias_hh_array = init.rand(hidden_size, device=device, dtype=dtype)
            self.bias_hh = Parameter(bias_hh_array, device=device, dtype=dtype)
        else:
            self.bias_ih = None
            self.bias_hh = None
        
        self.nonlinearity = nonlinearity
        ### END YOUR SOLUTION

    def forward(self, X, h=None):
        """
        Inputs:
        X of shape (bs, input_size): Tensor containing input features
        h of shape (bs, hidden_size): Tensor containing the initial hidden state
            for each element in the batch. Defaults to zero if not provided.

        Outputs:
        h' of shape (bs, hidden_size): Tensor contianing the next hidden state
            for each element in the batch.
        """
        ### BEGIN YOUR SOLUTION
        # raise NotImplementedError()
        temp1 = ops.matmul(X, self.W_ih)  # (bs, hidden_size) = (bs, input_size) * (input_size, hidden_size)
        if self.bias_ih is not None: # (hidden_size, ) -> (1, hidden_size) -> (bs, hidden_size)
            temp2 = self.bias_ih.reshape((1, self.bias_ih.shape[0])).broadcast_to(temp1.shape)
        else:
            temp2 = 0
        temp3 = temp1 + temp2

        if h is not None:
            temp4 = ops.matmul(h, self.W_hh)
        else:
            temp4 = 0

        if self.bias_hh is not None:
            temp5 = self.bias_hh.reshape((1, self.bias_hh.shape[0])).broadcast_to(temp1.shape)
        else:
            temp5 = 0

        temp6 = temp4 + temp5

        if self.nonlinearity == 'tanh':
            result = ops.tanh(temp3 + temp6)
        else:
            result = ops.relu(temp3 + temp6)
        return result
        ### END YOUR SOLUTION


class RNN(Module):
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True, nonlinearity='tanh', device=None, dtype="float32"):
        """
        Applies a multi-layer RNN with tanh or ReLU non-linearity to an input sequence.

        Parameters:
        input_size - The number of expected features in the input x
        hidden_size - The number of features in the hidden state h
        num_layers - Number of recurrent layers.
        nonlinearity - The non-linearity to use. Can be either 'tanh' or 'relu'.
        bias - If False, then the layer does not use bias weights.

        Variables:
        rnn_cells[k].W_ih: The learnable input-hidden weights of the k-th layer,
            of shape (input_size, hidden_size) for k=0. Otherwise the shape is
            (hidden_size, hidden_size).
        rnn_cells[k].W_hh: The learnable hidden-hidden weights of the k-th layer,
            of shape (hidden_size, hidden_size).
        rnn_cells[k].bias_ih: The learnable input-hidden bias of the k-th layer,
            of shape (hidden_size,).
        rnn_cells[k].bias_hh: The learnable hidden-hidden bias of the k-th layer,
            of shape (hidden_size,).
        """
        super().__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.rnn_cells = []

        for layer in range(self.num_layers):
            layer_input_size = input_size if layer == 0 else hidden_size
            cell = RNNCell(layer_input_size, hidden_size, bias=bias, nonlinearity=nonlinearity, device=device, dtype=dtype)
            self.rnn_cells.append(cell)

    def forward(self, X, h0=None):
        """
        Inputs:
        X of shape (seq_len, bs, input_size) containing the features of the input sequence.
        h_0 of shape (num_layers, bs, hidden_size) containing the initial
            hidden state for each element in the batch. Defaults to zeros if not provided.

        Outputs
        output of shape (seq_len, bs, hidden_size) containing the output features
            (h_t) from the last layer of the RNN, for each t.
        h_n of shape (num_layers, bs, hidden_size) containing the final hidden state for each element in the batch.
        """
        ### BEGIN YOUR SOLUTION
        seq_len, bs, _ = X.shape

        if h0 is None:
            h0 = init.zeros(self.num_layers, bs, self.hidden_size, device=X.device, dtype=X.dtype)
            h0_split = ops.split(h0, axis=0)
        else:
            h0_split = ops.split(h0, axis=0)  # a tuple could subscript

        hn = []
        output = X
        for layer in range(self.num_layers):
            h = h0_split[layer]
            layer_out = []

            out_split = ops.split(output, axis=0)
            for t in range(seq_len):
                h = self.rnn_cells[layer].forward(out_split[t], h)
                layer_out.append(h)
            
            output = ops.stack(layer_out, axis=0)
            hn.append(h)
    
        hn = ops.stack(hn, axis=0)
        return output, hn
        ### END YOUR SOLUTION


class LSTMCell(Module):
    def __init__(self, input_size, hidden_size, bias=True, device=None, dtype="float32"):
        """
        A long short-term memory (LSTM) cell.

        Parameters:
        input_size - The number of expected features in the input X
        hidden_size - The number of features in the hidden state h
        bias - If False, then the layer does not use bias weights

        Variables:
        W_ih - The learnable input-hidden weights, of shape (input_size, 4*hidden_size).
        W_hh - The learnable hidden-hidden weights, of shape (hidden_size, 4*hidden_size).
        bias_ih - The learnable input-hidden bias, of shape (4*hidden_size,).
        bias_hh - The learnable hidden-hidden bias, of shape (4*hidden_size,).

        Weights and biases are initialized from U(-sqrt(k), sqrt(k)) where k = 1/hidden_size
        """
        super().__init__()
        ### BEGIN YOUR SOLUTION
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.device = device
        self.dtype = dtype

        k = 1 / self.hidden_size
        bound = (k)**0.5

        # hidden_size = 4*hidden_size

        W_ih_array = init.rand(input_size, 4*hidden_size, low=-bound, high=bound, device=device, dtype=dtype)
        self.W_ih = Parameter(W_ih_array, device=device, dtype=dtype)

        W_hh_array = init.rand(hidden_size, 4*hidden_size, low=-bound, high=bound, device=device, dtype=dtype)
        self.W_hh = Parameter(W_hh_array, device=device, dtype=dtype)

        if bias:
            bias_ih_array = init.rand(4*hidden_size, device=device, dtype=dtype)
            self.bias_ih= Parameter(bias_ih_array, device=device, dtype=dtype)

            bias_hh_array = init.rand(4*hidden_size, device=device, dtype=dtype)
            self.bias_hh = Parameter(bias_hh_array, device=device, dtype=dtype)
        else:
            self.bias_ih = None
            self.bias_hh = None
        ### END YOUR SOLUTION


    def forward(self, X, h=None):
        """
        Inputs: X, h
        X of shape (batch, input_size): Tensor containing input features
        h, tuple of (h0, c0), with
            h0 of shape (bs, hidden_size): Tensor containing the initial hidden state
                for each element in the batch. Defaults to zero if not provided.
            c0 of shape (bs, hidden_size): Tensor containing the initial cell state
                for each element in the batch. Defaults to zero if not provided.

        Outputs: (h', c')
        h' of shape (bs, hidden_size): Tensor containing the next hidden state for each
            element in the batch.
        c' of shape (bs, hidden_size): Tensor containing the next cell state for each
            element in the batch.
        """
        ### BEGIN YOUR SOLUTION
        temp1 = ops.matmul(X, self.W_ih)  # (bs, 4*hidden_size) = (bs, input_size) * (input_size, 4*hidden_size)

        if h is not None:
            h0, c0 = h
            temp2 = ops.matmul(h0, self.W_hh)
        else:
            h0, c0 = 0, 0
            temp2 = 0

        if self.bias:
            temp3 = self.bias_ih.reshape((1, self.bias_ih.shape[0])).broadcast_to(temp1.shape)
            temp4 = self.bias_hh.reshape((1, self.bias_hh.shape[0])).broadcast_to(temp1.shape)
        else:
            temp3 = 0
            temp4 = 0
        
        temp5 = temp1 + temp2 + temp3 + temp4  # (bs, 4*hidden_size)

        temp_tuple = ops.split(temp5, axis=1)  # temp_tuple is a TensorTuple type, can only index 1 elem in it once
        length = len(temp_tuple) // 4
        i_list, f_list, g_list, o_list = [], [], [], []
        for i in range(length):
            i_list.append(temp_tuple[i])
            f_list.append(temp_tuple[i + length])
            g_list.append(temp_tuple[i + length * 2])
            o_list.append(temp_tuple[i + length * 3])
        i = ops.stack(i_list, axis=1)
        f = ops.stack(f_list, axis=1)
        g = ops.stack(g_list, axis=1)
        o = ops.stack(o_list, axis=1)

        i, f, g, o = sigmoid(i), sigmoid(f), ops.tanh(g), sigmoid(o)

        c_out = f * c0 + i * g
        h_out = o * ops.tanh(c_out)

        return h_out, c_out
        ### END YOUR SOLUTION


class LSTM(Module):
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True, device=None, dtype="float32"):
        super().__init__()
        """
        Applies a multi-layer long short-term memory (LSTM) RNN to an input sequence.

        Parameters:
        input_size - The number of expected features in the input x
        hidden_size - The number of features in the hidden state h
        num_layers - Number of recurrent layers.
        bias - If False, then the layer does not use bias weights.

        Variables:
        lstm_cells[k].W_ih: The learnable input-hidden weights of the k-th layer,
            of shape (input_size, 4*hidden_size) for k=0. Otherwise the shape is
            (hidden_size, 4*hidden_size).
        lstm_cells[k].W_hh: The learnable hidden-hidden weights of the k-th layer,
            of shape (hidden_size, 4*hidden_size).
        lstm_cells[k].bias_ih: The learnable input-hidden bias of the k-th layer,
            of shape (4*hidden_size,).
        lstm_cells[k].bias_hh: The learnable hidden-hidden bias of the k-th layer,
            of shape (4*hidden_size,).
        """
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.device = device
        self.dtype = dtype
        self.lstm_cells = []

        for layer in range(self.num_layers):
            layer_input_size = self.input_size if layer == 0 else self.hidden_size
            cell = LSTMCell(layer_input_size, self.hidden_size, self.bias, self.device, self.dtype)
            self.lstm_cells.append(cell)

    def forward(self, X, h=None):
        """
        Inputs: X, h
        X of shape (seq_len, bs, input_size) containing the features of the input sequence.
        h, tuple of (h0, c0) with
            h_0 of shape (num_layers, bs, hidden_size) containing the initial
                hidden state for each element in the batch. Defaults to zeros if not provided.
            c0 of shape (num_layers, bs, hidden_size) containing the initial
                hidden cell state for each element in the batch. Defaults to zeros if not provided.

        Outputs: (output, (h_n, c_n))
        output of shape (seq_len, bs, hidden_size) containing the output features
            (h_t) from the last layer of the LSTM, for each t.
        tuple of (h_n, c_n) with
            h_n of shape (num_layers, bs, hidden_size) containing the final hidden state for each element in the batch.
            h_n of shape (num_layers, bs, hidden_size) containing the final hidden cell state for each element in the batch.
        """
        ### BEGIN YOUR SOLUTION
        seq_len, bs, _ = X.shape

        if h is None:
            h0 = init.zeros(self.num_layers, bs, self.hidden_size, device=X.device, dtype=X.dtype)
            c0 = init.zeros(self.num_layers, bs, self.hidden_size, device=X.device, dtype=X.dtype)
            h0_split = ops.split(h0, axis=0)
            c0_split = ops.split(c0, axis=0)
        else:
            h0, c0 = h  # h0 c0 is (num_layers, bs, hidden_size)
            h0_split = ops.split(h0, axis=0)  # h0 c0 is (bs, hidden_size)
            c0_split = ops.split(c0, axis=0)

        hn, cn= [], []
        output = X
        for layer in range(self.num_layers):
            h, c = h0_split[layer], c0_split[layer]
            layer_out = []

            out_split = ops.split(output, axis=0)  # (bs, input_size)
            for t in range(seq_len):
                h, c = self.lstm_cells[layer](out_split[t], (h, c))
                layer_out.append(h)

            output = ops.stack(layer_out, axis=0)
            hn.append(h)
            cn.append(c)

        hn = ops.stack(hn, axis=0)
        cn = ops.stack(cn, axis=0)
        return output, (hn, cn)
        ### END YOUR SOLUTION

class Embedding(Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype="float32"):
        super().__init__()
        """
        Maps one-hot word vectors from a dictionary of fixed size to embeddings.

        Parameters:
        num_embeddings (int) - Size of the dictionary
        embedding_dim (int) - The size of each embedding vector

        Variables:
        weight - The learnable weights of shape (num_embeddings, embedding_dim)
            initialized from N(0, 1).
        """
        ### BEGIN YOUR SOLUTION
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        weight_array = init.randn(num_embeddings, embedding_dim, device=device, dtype=dtype)
        self.weight = Parameter(weight_array, device=device, dtype=dtype)
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        """
        Maps word indices to one-hot vectors, and projects to embedding vectors

        Input:
        x of shape (seq_len, bs)

        Output:
        output of shape (seq_len, bs, embedding_dim)
        """
        ### BEGIN YOUR SOLUTION
        # (seq_len, bs) --represent as one hot--> (seq_len, bs, num_embeddings) --mul self.weight--> (seq_len, bs, embedding_dim)
        # but for efficiency, we can not use mul, instead use indexing

        # can not use index to get weight, because weight is a parameter, the grad will not pass to it.
        seq_len, bs = x.shape
        num_embeddings, embedding_dim = self.weight.shape  # num_embeddings = vocab_size, means the number of words, length of dictionary, embedding_dim means the input size
        ll = []
        idx_np_array = x.cached_data.numpy()  # if directly index x.cache_data will get a NDArray with 0 dim, not an int type
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                idx = int(idx_np_array[i, j])
                # vec = init.zeros(self.num_embeddings, device=self.weight.device, dtype=self.weight.dtype)  # (num_embeddings, )
                # vec = numpy.zeros()
                vec = self.weight.device.full((self.num_embeddings, ), 0, dtype=self.weight.dtype)
                vec[idx] = 1
                one_hot = Tensor(vec, device=self.weight.device, dtype=self.weight.dtype)
                # vec = Tensor(self.weight.cached_data[idx, :], device=self.weight.device, dtype=self.weight.dtype)
                ll.append(one_hot)
        temp1 = ops.stack(ll, axis=0)  # (seq_len*bs, num_embeddings)
        temp2 = temp1 @ self.weight  # (seq_len*bs, num_embeddings) @ (num_embeddings, embedding_dim) = (seq_len*bs, embedding_dim)
        temp3 = ops.reshape(temp2, (seq_len, bs, embedding_dim))
        return temp3
        ### END YOUR SOLUTION
