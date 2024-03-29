"""hw1/apps/simple_ml.py"""

import struct
import gzip
import numpy as np

import sys

sys.path.append("python/")
import needle as ndl


def parse_mnist(image_filesname, label_filename):
    """Read an images and labels file in MNIST format.  See this page:
    http://yann.lecun.com/exdb/mnist/ for a description of the file format.

    Args:
        image_filename (str): name of gzipped images file in MNIST format
        label_filename (str): name of gzipped labels file in MNIST format

    Returns:
        Tuple (X,y):
            X (numpy.ndarray[np.float32]): 2D numpy array containing the loaded
                data.  The dimensionality of the data should be
                (num_examples x input_dim) where 'input_dim' is the full
                dimension of the data, e.g., since MNIST images are 28x28, it
                will be 784.  Values should be of type np.float32, and the data
                should be normalized to have a minimum value of 0.0 and a
                maximum value of 1.0.

            y (numpy.ndarray[dypte=np.int8]): 1D numpy array containing the
                labels of the examples.  Values should be of type np.int8 and
                for MNIST will contain the values 0-9.
    """
    ### BEGIN YOUR SOLUTION
    with gzip.open(image_filesname, 'rb') as f:
        magic_number = int.from_bytes(f.read(4), 'big')
        num_images = int.from_bytes(f.read(4), 'big')
        num_rows = int.from_bytes(f.read(4), 'big')
        num_cols = int.from_bytes(f.read(4), 'big')
        images = np.frombuffer(f.read(), dtype=np.uint8).reshape(num_images, num_rows * num_cols)

    with gzip.open(label_filename, 'rb') as f:
        magic_number = int.from_bytes(f.read(4), 'big')
        num_labels = int.from_bytes(f.read(4), 'big')
        labels = np.frombuffer(f.read(), dtype=np.uint8)

    X = images.astype(np.float32) / 255.0
    y = labels.astype(np.uint8)

    return X, y
    ### END YOUR SOLUTION


def softmax_loss(Z, y_one_hot):
    """Return softmax loss.  Note that for the purposes of this assignment,
    you don't need to worry about "nicely" scaling the numerical properties
    of the log-sum-exp computation, but can just compute this directly.

    Args:
        Z (ndl.Tensor[np.float32]): 2D Tensor of shape
            (batch_size, num_classes), containing the logit predictions for
            each class.
        y_one_hot (ndl.Tensor[np.int8]): 2D Tensor of shape (batch_size, num_classes)
            containing a 1 at the index of the true label of each example and
            zeros elsewhere.

    Returns:
        Average softmax loss over the sample. (ndl.Tensor[np.float32])
    """
    # Number of samples
    N = Z.shape[0]

    # Compute the softmax
    # By subtracting the maximum value from each element along the rows,
    # we ensure numerical stability and prevent overflow issues when computing the softmax probabilities,
    # making the computation more robust and accurate.
    exp_Z = ndl.exp(Z)
    exp_sum = ndl.broadcast_to(ndl.reshape(ndl.summation(exp_Z, axes=1), (exp_Z.shape[0], 1)), exp_Z.shape)
    softmax_scores = exp_Z / exp_sum  # shape N * k

    # Cross-entropy loss
    # Note: we use np.log(x + eps) to avoid log(0) which is undefined.
    # eps = 1e-10
    true_score = softmax_scores * y_one_hot
    true_score_1 = ndl.summation(true_score, axes=1)
    cross_entropy = -ndl.log(true_score_1)  # shape N * 1

    # Compute average loss
    average_loss = ndl.summation(cross_entropy, axes=0) / N  # shape 1 * 1

    return average_loss
 

def nn_epoch(X, y, W1, W2, lr=0.1, batch=100):
    """Run a single epoch of SGD for a two-layer neural network defined by the
    weights W1 and W2 (with no bias terms):
        logits = ReLU(X * W1) * W1
    The function should use the step size lr, and the specified batch size (and
    again, without randomizing the order of X).

    Args:
        X (np.ndarray[np.float32]): 2D input array of size
            (num_examples x input_dim).
        y (np.ndarray[np.uint8]): 1D class label array of size (num_examples,)
        W1 (ndl.Tensor[np.float32]): 2D array of first layer weights, of shape
            (input_dim, hidden_dim)
        W2 (ndl.Tensor[np.float32]): 2D array of second layer weights, of shape
            (hidden_dim, num_classes)
        lr (float): step size (learning rate) for SGD
        batch (int): size of SGD mini-batch

    Returns:
        Tuple: (W1, W2)
            W1: ndl.Tensor[np.float32]
            W2: ndl.Tensor[np.float32]
    """

    num_examples = X.shape[0]
    # print("total epoch: ", num_examples/batch)

    for i in range(0, num_examples, batch):
        # print("epoch: ", i)
        x_batch = ndl.Tensor(X[i:i+batch])
        y_batch = y[i:i+batch]

        temp = np.zeros((batch, np.max(y_batch)+1))
        temp[range(batch), y_batch] = 1
        y_one_hot = ndl.Tensor(temp)

        z1 = x_batch
        z2 = ndl.relu(ndl.matmul(z1, W1))
        z3 = ndl.matmul(z2, W2)

        loss = softmax_loss(z3, y_one_hot)
        loss.backward()

        W2 -= lr * W2.grad
        W1 -= lr * W1.grad
    
    return W1, W2


### CODE BELOW IS FOR ILLUSTRATION, YOU DO NOT NEED TO EDIT


def loss_err(h, y):
    """Helper function to compute both loss and error"""
    y_one_hot = np.zeros((y.shape[0], h.shape[-1]))
    y_one_hot[np.arange(y.size), y] = 1
    y_ = ndl.Tensor(y_one_hot)
    return softmax_loss(h, y_).numpy(), np.mean(h.numpy().argmax(axis=1) != y)
