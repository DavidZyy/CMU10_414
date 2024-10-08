from typing import List
from needle.autograd import Tensor
import needle.backend_ndarray.ndarray as ndarray
from needle import ops
import needle.init as init
import numpy as np
from .nn_sequence import Embedding
from .nn_basic import (
    Parameter, 
    Module, 
    ReLU,
    Dropout,
    LayerNorm1d,
    Linear,
    Sequential
)


class MultiHeadAttention(Module):
    """
    The multi-head self attention module.
    """
    def __init__(
        self,
        *,
        dropout = 0.,
        causal = False,
        device = None,
        dtype = "float32",
    ):

        super().__init__()

        self.device = device
        self.dtype = dtype

        self.causal = causal
        self.dropout = Dropout(dropout)

    def create_causal_mask(self, i, j, device):
        """
        return a triangular causal mask.
        """
        mask = -np.finfo(np.float32).max * np.triu(
            np.ones((1, 1, i, j), dtype=np.float32), j - i + 1)

        return ndarray.array(
            mask, device=device)

    def matmul(self, a: Tensor, b_transpose: Tensor):
        """
        batched matrix multiplication;
        """
        # see my typora note of this batched matmul algorithm
        a_new_shape = (*a.shape, 1)
        b_new_shape = (*b_transpose.shape[:-2], 1, b_transpose.shape[-2], b_transpose.shape[-1])

        a.cached_data = a.cached_data.compact()
        b_transpose.cached_data = b_transpose.cached_data.compact()
        a_reshaped = a.reshape(a_new_shape)
        b_reshaped = b_transpose.reshape(b_new_shape)

        broadcast_shape = (*a.shape, b_transpose.shape[-1])

        a_broadcast = a_reshaped.broadcast_to(broadcast_shape)
        b_broadcast = b_reshaped.broadcast_to(broadcast_shape)

        out = (a_broadcast * b_broadcast).sum(len(broadcast_shape) - 2)

        return out

    def softmax(self, logit):
        """
        The softmax function; 
        """
        max_val = Tensor(
            logit.realize_cached_data().max(axis=3),
            device=logit.device,
            dtype=logit.dtype,
            requires_grad=False
        )

        max_val = max_val.reshape((*logit.shape[:-1], 1))
        max_val = max_val.broadcast_to(logit.shape)

        probs = ops.exp(logit - max_val)

        denom = probs.sum(axes=3)
        denom = denom.reshape((*logit.shape[:-1], 1))
        denom = denom.broadcast_to(logit.shape)

        return probs / denom

    def forward(
        self,
        q, k, v,
    ):
        """
        The forward function of the MultiHeadAttention activation function.
        Input: three states q, k, v, with shape (batch_size, num_head, seq_len, dim_head)
        Output: the activation output `result` and attention softmax probability `probs` (with dropout applied)

        zyy: all tensor have the shape of (B, H, T, D), 
        formula: out = softmax(q @ k.T / sqrt(dim_head)) @ v
        q (B, H, T, D) @ k.T (B, H, D, T) @ v (B, H, T, D) = X (B, H, T, D)
        """
        B, H, T, D = q.shape
        batch_size, num_head, queries_len, q_dim = q.shape
        _, _, keys_values_len, k_dim = k.shape
        _, _, _, v_dim = v.shape

        assert q_dim == k_dim == v_dim

        ### BEGIN YOUR SOLUTION
        k_T = ops.transpose(k, axes=(2, 3))
        k_T.cached_data = k_T.cached_data.compact()  # for reshape
        mask = self.create_causal_mask(T, T, q.device)
        Tmask = Tensor(mask, dtype=q.dtype, device=q.device)  # (1, 1, T, T)
        shape = (B, H, T, T)
        Tmask = ops.broadcast_to(Tmask, shape)  # (B, H, T, T)

        temp1 = self.matmul(q, k_T)  # (B, H, T, T)
        temp2 = temp1 / q_dim**0.5  # (B, H, T, T)
        if self.causal:
            temp3 = temp2 + Tmask  # (B, H, T, T)
        else:
            temp3 = temp2
        temp4 = self.softmax(temp3)  # (B, H, T, T)
        temp5 = self.dropout(temp4)  # (B, H, T, T)
        temp6 = self.matmul(temp5, v)  # (B, H, T, D)

        result = temp6
        probs = temp5
        ### END YOUR SOLUTION

        return result, probs


class AttentionLayer(Module):

    def __init__(
        self,
        q_features: int,
        num_head: int,
        dim_head: int,
        *,
        k_features: int = None,
        v_features: int = None,
        out_features: int = None,
        dropout = 0.,
        causal = True,
        device = None,
        dtype = "float32",
    ):

        super().__init__()

        self.device = device
        self.dtype = dtype

        if k_features is None:
            k_features = q_features
        if v_features is None:
            v_features = q_features
        if out_features is None:
            out_features = q_features

        self.q_features = q_features
        self.k_features = k_features
        self.v_features = v_features
        self.out_features = out_features

        self.num_head = num_head
        self.dim_head = dim_head

        self.prenorm_q = LayerNorm1d(
            q_features, device=device, dtype=dtype)
        self.prenorm_k = LayerNorm1d(
            k_features, device=device, dtype=dtype)
        self.prenorm_v = LayerNorm1d(
            v_features, device=device, dtype=dtype)

        inner_dim = num_head * dim_head
        
        self.q_projection = Linear(
            q_features, inner_dim, bias=False,
            device=device, dtype=dtype)
        self.k_projection = Linear(
            k_features, inner_dim, bias=False,
            device=device, dtype=dtype)
        self.v_projection = Linear(
            v_features, inner_dim, bias=False,
            device=device, dtype=dtype)

        self.attn = MultiHeadAttention(
            dropout=dropout, causal=causal,
            device=device, dtype=dtype)

        self.out_projection = Linear(
            inner_dim, out_features, bias=False,
            device=device, dtype=dtype)

    def forward(
        self,
        q, k=None, v=None,
    ):
        """
        The forward function of the self-attention layer.
        Input: `q` with shape (batch_size, q_len, q_dim)
               `k` (if not None) with shape (batch_size, kv_len, k_dim)
               `v` (if not None) with shape (batch_size, kv_len, v_dim)
        Output: the output `result` with shape (batch_size, kv_len, out_features)
        """

        if k is None:
            k = q
        if v is None:
            v = q

        B, T, D0 = q.shape
        batch_size, queries_len, q_dim = q.shape
        _, keys_values_len, k_dim = k.shape
        _, _, v_dim = v.shape

        result = None

        ### BEGIN YOUR SOLUTION
        q = ops.reshape(q, (q.shape[0]*q.shape[1], q.shape[2]))  # (B, T, D0) -> (B*T, D0)
        k = ops.reshape(k, (k.shape[0]*k.shape[1], k.shape[2]))
        v = ops.reshape(v, (v.shape[0]*v.shape[1], v.shape[2]))

        Qn = self.prenorm_q(q)  # (B*T, D0)
        Kn = self.prenorm_k(k)
        Vn = self.prenorm_v(v)

        Qw = self.q_projection(Qn)  # (B*T, HD) = (B*T, D0) @ (D0, HD)  H means num_head, D means dim_head
        Kw = self.k_projection(Kn)
        Vw = self.v_projection(Vn)

        Qu = ops.reshape(Qw, (B, T, self.num_head, self.dim_head))  # (B, T, H, D)
        Ku = ops.reshape(Kw, (B, keys_values_len, self.num_head, self.dim_head))
        Vu = ops.reshape(Vw, (B, keys_values_len, self.num_head, self.dim_head))

        Qu = ops.transpose(Qu, axes=(2, 1))  # (B, H, T, D)
        Ku = ops.transpose(Ku, axes=(2, 1))
        Vu = ops.transpose(Vu, axes=(2, 1))

        result, probs = self.attn(Qu, Ku, Vu)  # result: (B, H, T, D)

        result = ops.transpose(result, axes=(2, 1))  # (B, T, H, D)
        result.cached_data = result.cached_data.compact()
        result = ops.reshape(result, (B*T, self.num_head*self.dim_head))  # (B*T, H*D)

        result = self.out_projection(result)  # (B*T, out_features)
        result = ops.reshape(result, (B, T, result.shape[1]))  # (B, T, out_features)
        ### END YOUR SOLUTION

        return result


class TransformerLayer(Module):

    def __init__(
        self,
        q_features: int,
        num_head: int,
        dim_head: int,
        hidden_size: int,
        *,
        dropout = 0.,
        causal = True,
        device = None,
        dtype = "float32",
    ):

        super().__init__()

        self.device = device
        self.dtype = dtype

        ### BEGIN YOUR SOLUTION
        # raise NotImplementedError()
        self.attn = AttentionLayer(
            q_features=q_features,
            num_head=num_head,
            dim_head=dim_head,

            dropout=dropout,
            causal=causal,
            device=device,
            dtype=dtype)
        self.dropout = Dropout(dropout)
        self.layernorm = LayerNorm1d(
            q_features, device=device, dtype=dtype)
        # self.linear1 = Linear(
        #     q_features, hidden_size, bias=False,
        #     device=device, dtype=dtype)
        # self.linear2 = Linear(
        #     hidden_size, q_features, bias=False,
        #     device=device, dtype=dtype)
        # the bias should be used
        self.linear1 = Linear(
            q_features, hidden_size,
            device=device, dtype=dtype)
        self.linear2 = Linear(
            hidden_size, q_features,
            device=device, dtype=dtype)
        self.relu = ReLU()
        ### END YOUR SOLUTION

    def forward(
        self,
        x
    ):
        """
        The forward function of a Transformer Layer.
        Input: the hidden states from previous layers `x` with shape (batch_size, seq_len, x_dim)
        Ouput: the hidden states after the Transformer Layer `x` with shape (batch_size, seq_len, x_dim)
        """

        batch_size, seq_len, x_dim = x.shape

        # x = x + self.dropout(self.attn(x))
        temp0 = self.attn(x)  # (B, T, D) -> (B, T, D)
        temp1 = self.dropout(temp0)  # (B, T, D)
        temp2 = x + temp1  # (B, T, D)
        
        # x = x + self.dropout(self.linear2(self.dropout(self.relu(self.linear1(self.layernorm(x))))))
        temp3 = ops.reshape(temp2, (batch_size*seq_len, x_dim))  # (B*T, D)
        temp4 = self.layernorm(temp3)  # (B*T, D)
        temp5 = self.linear1(temp4)  # (B*T, hidden_size)
        temp6 = self.relu(temp5)
        temp7 = self.dropout(temp6)
        temp8 = self.linear2(temp7)  # (B*T, D)
        temp9 = self.dropout(temp8)
        temp10 = temp3 + temp9
        temp11 = ops.reshape(temp10, (batch_size, seq_len, x_dim))
        
        return temp11

        # x = x + self.dropout(self.attn(x))  # (B, T, D)
        # x = ops.reshape(x, (batch_size*seq_len, x_dim))  # (B*T, D)
        # x = x + self.dropout(self.linear2(self.dropout(self.relu(self.linear1(self.layernorm(x))))))  # (B*T, D)
        # x = ops.reshape(x, (batch_size, seq_len, x_dim))  # (B, T, D)
        # return x

        


class Transformer(Module):

    def __init__(
        self,
        embedding_size: int,
        hidden_size: int,
        num_layers: int, 
        *,
        num_head: int = 8,
        dim_head: int = 32,
        dropout = 0.,
        causal = True,
        device = None,
        dtype = "float32",
        batch_first = False,
        sequence_len = 2048
    ):

        super().__init__()

        self.device = device
        self.dtype = dtype
        self.batch_first = batch_first

        ### BEGIN YOUR SOLUTION
        self.transformer_layers = Sequential(
            *[TransformerLayer(
                q_features=embedding_size,
                num_head=num_head,
                dim_head=dim_head,
                hidden_size=hidden_size,
                dropout=dropout,
                causal=causal,
                device=device,
                dtype=dtype)
                for _ in range(num_layers)])

        self.embedding = Embedding(
            num_embeddings=sequence_len,
            embedding_dim=embedding_size,
            device=device,
            dtype=dtype)
        ### END YOUR SOLUTION

    def forward(
        self,
        x, h=None
    ):
        """
        Input:
            x: input features with shape (seq_len, batch_size, embedding_size): emdedding_size means the input x is pass in from the embedding layer
            h: hidden state with shape (num_layers, batch_size, hidden_size)
        Output:
            out: hidden states with shape (seq_len, batch_size, hidden_size)
            h_out: hidden states with shape (num_layers, batch_size, hidden_size)
        """
        if not self.batch_first:
            x = ops.transpose(x, axes=(0, 1))  # x (batch_size, seq_len, embedding_size)

        ### BEGIN YOUR SOLUTION
        batch_size, seq_len, embedding_size = x.shape
        position_array = np.arange(seq_len).reshape((seq_len, 1))
        position = Tensor(position_array, device=x.device, dtype=x.dtype)  # (seq_len, 1)
        position = ops.broadcast_to(position, (seq_len, batch_size))  # (seq_len, batch_size)
        pos_emb = self.embedding(position)  # (seq_len, batch_size, embedding_size)
        pos_emb = ops.transpose(pos_emb, axes=(1, 0))  # (batch_size, seq_len, embedding_size)
        x = x + pos_emb
        x = self.transformer_layers(x)
        ### END YOUR SOLUTION

        if not self.batch_first:
            x = ops.transpose(x, axes=(0, 1))

        return x, init.zeros_like(x)