import sys
sys.path.append('./python')
import needle as ndl
import needle.nn as nn
import math
import numpy as np
np.random.seed(0)


def ResidualBlock(in_channels, out_channels, kernel_size, stride, device=None, dtype="float32"):
    main_path = nn.Sequential(nn.ConvBN(in_channels, out_channels, kernel_size, stride, device=device, dtype=dtype),
                              nn.ConvBN(in_channels, out_channels, kernel_size, stride, device=device, dtype=dtype))

    return nn.Residual(main_path)

class ResNet9(ndl.nn.Module):
    def __init__(self, device=None, dtype="float32"):
        super().__init__()
        # zyy:
        self.device = device
        ### BEGIN YOUR SOLUTION ###
        self.model = nn.Sequential(nn.ConvBN(3, 16, 7, 4, device=device, dtype=dtype),
                                   nn.ConvBN(16, 32, 3, 2, device=device, dtype=dtype),
                                   ResidualBlock(32, 32, 3, 1, device=device, dtype=dtype),

                                   nn.ConvBN(32, 64, 3, 2, device=device, dtype=dtype),
                                   nn.ConvBN(64, 128, 3, 2, device=device, dtype=dtype),
                                   ResidualBlock(128, 128,3, 1, device=device, dtype=dtype),

                                   nn.Flatten(),

                                   nn.Linear(128, 128, device=device, dtype=dtype),
                                   nn.ReLU(),
                                   nn.Linear(128, 10, device=device, dtype=dtype))
        ### END YOUR SOLUTION

    def forward(self, x):
        ### BEGIN YOUR SOLUTION
        # raise NotImplementedError()
        return self.model(x)
        ### END YOUR SOLUTION


class LanguageModel(nn.Module):
    def __init__(self, embedding_size, output_size, hidden_size, num_layers=1,
                 seq_model='rnn', device=None, dtype="float32"):
        """
        Consists of an embedding layer, a sequence model (either RNN or LSTM), and a
        linear layer.
        Parameters:
        output_size: Size of dictionary
        embedding_size: Size of embeddings
        hidden_size: The number of features in the hidden state of LSTM or RNN
        seq_model: 'rnn' or 'lstm', whether to use RNN or LSTM
        num_layers: Number of layers in RNN or LSTM
        """
        super(LanguageModel, self).__init__()
        ### BEGIN YOUR SOLUTION
        self.embedding_size = embedding_size 
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.seq_model = seq_model
        self.device = device
        self.dtype = dtype
        self.embedding = nn.Embedding(self.output_size, self.embedding_size, device=self.device, dtype=self.dtype) 
        if self.seq_model == "lstm":
            # embedding_size act as input_sizie here
            self.model = nn.LSTM(self.embedding_size, self.hidden_size, self.num_layers, device=self.device, dtype=self.dtype)
        else:
            self.model = nn.RNN(self.embedding_size, self.hidden_size, self.num_layers, device=self.device, dtype=self.dtype)
        self.linear = nn.Linear(self.hidden_size, self.output_size, device=self.device, dtype=self.dtype)
        ### END YOUR SOLUTION

    def forward(self, x, h=None):
        """
        Given sequence (and the previous hidden state if given), returns probabilities of next word
        (along with the last hidden state from the sequence model).
        Inputs:
        x of shape (seq_len, bs)
        h of shape (num_layers, bs, hidden_size) if using RNN,
            else h is tuple of (h0, c0), each of shape (num_layers, bs, hidden_size)
        Returns (out, h)
        out of shape (seq_len*bs, output_size)
        h of shape (num_layers, bs, hidden_size) if using RNN,
            else h is tuple of (h0, c0), each of shape (num_layers, bs, hidden_size)
        """
        ### BEGIN YOUR SOLUTION
        # raise NotImplementedError()
        temp1 = self.embedding(x)  # (seq_len, bs, emb_size) or (seq_len, bs, input_size)
        output, h = self.model(temp1, h)  # output is (seq_len, bs, output_size), h is (num_layers, bs, hidden_size)
        temp2 = ndl.ops.reshape(output, (output.shape[0]*output.shape[1], output.shape[2]))  # (seq_len*bs, output_size)
        temp3 = self.linear(temp2)  # (seq_len*bs, output_size)
        return temp3, h
        ### END YOUR SOLUTION


if __name__ == "__main__":
    model = ResNet9()
    x = ndl.ops.randu((1, 32, 32, 3), requires_grad=True)
    model(x)
    cifar10_train_dataset = ndl.data.CIFAR10Dataset("data/cifar-10-batches-py", train=True)
    train_loader = ndl.data.DataLoader(cifar10_train_dataset, 128, ndl.cpu(), dtype="float32")
    print(dataset[1][0].shape)