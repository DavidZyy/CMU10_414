import sys

sys.path.append("../python")
import needle as ndl
import needle.nn as nn
import numpy as np
import time
import os

np.random.seed(0)
# MY_DEVICE = ndl.backend_selection.cuda()


def ResidualBlock(dim, hidden_dim, norm=nn.BatchNorm1d, drop_prob=0.1):
    ### BEGIN YOUR SOLUTION
    main_path = nn.Sequential(
        nn.Linear(dim, hidden_dim),
        norm(hidden_dim),
        nn.ReLU(),
        nn.Dropout(drop_prob),
        nn.Linear(hidden_dim, dim),
        norm(dim)
    )
    res = nn.Residual(main_path)
    result = nn.Sequential(res, nn.ReLU())
    return result
    ### END YOUR SOLUTION
# def ResidualBlock(dim, hidden_dim, norm=nn.BatchNorm1d, drop_prob=0.1):
#     ### BEGIN YOUR SOLUTION
#     modules = nn.Sequential(
#         nn.Linear(dim, hidden_dim),
#         norm(hidden_dim),
#         nn.ReLU(),
#         nn.Dropout(drop_prob),
#         nn.Linear(hidden_dim, dim),
#         norm(dim)
#     )
#     return nn.Sequential(
#         nn.Residual(modules),
#         nn.ReLU()
#     )

def MLPResNet(
    dim,
    hidden_dim=100,
    num_blocks=3,
    num_classes=10,
    norm=nn.BatchNorm1d,
    drop_prob=0.1,
):
    ### BEGIN YOUR SOLUTION
    resnet = nn.Sequential(
        nn.Flatten(),
        nn.Linear(dim, hidden_dim),
        nn.ReLU(),
        *[ResidualBlock(dim=hidden_dim, hidden_dim=hidden_dim//2, norm=norm, drop_prob=drop_prob) for _ in range(num_blocks)],
        nn.Linear(hidden_dim, num_classes)
    )

    return resnet
    ### END YOUR SOLUTION


def epoch(dataloader, model, opt=None):
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    # raise NotImplementedError()
    tot_loss, tot_error = [], 0.0
    loss_fn = nn.SoftmaxLoss()
    if opt is None:
        model.eval()
        for X, y in dataloader:
            logits = model(X)
            loss = loss_fn(logits, y)
            tot_error += np.sum(logits.numpy().argmax(axis=1) != y.numpy())
            tot_loss.append(loss.numpy())
    else:
        model.train()
        for X, y in dataloader:
            logits = model(X)
            loss = loss_fn(logits, y)
            tot_error += np.sum(logits.numpy().argmax(axis=1) != y.numpy())
            tot_loss.append(loss.numpy())
            opt.reset_grad()
            loss.backward()
            opt.step()
    sample_nums = len(dataloader.dataset)
    return tot_error/sample_nums, np.mean(tot_loss)

    ### END YOUR SOLUTION


def train_mnist(
    batch_size=100,
    epochs=10,
    optimizer=ndl.optim.Adam,
    lr=0.001,
    weight_decay=0.001,
    hidden_dim=100,
    data_dir="data",
):
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    # raise NotImplementedError()
    model = MLPResNet(28 * 28, hidden_dim=hidden_dim, num_classes=10)

    opt = optimizer(model.parameters(), lr=lr, weight_decay=weight_decay)

    mnist_train_dataset = ndl.data.MNISTDataset(
        data_dir+"train-images-idx3-ubyte.gz", data_dir+"train-labels-idx1-ubyte.gz"
    )
    mnist_train_dataloader = ndl.data.DataLoader(
        dataset=mnist_train_dataset, batch_size=batch_size, shuffle=True
    )
    mnist_test_dataset = ndl.data.MNISTDataset(
        data_dir+"t10k-images-idx3-ubyte.gz", data_dir+"t10k-labels-idx1-ubyte.gz"
    )
    mnist_test_dataloader = ndl.data.DataLoader(
        dataset=mnist_test_dataset, batch_size=batch_size
    )
    for _ in range(epochs):
        train_err, train_loss = epoch(mnist_train_dataloader, model, opt)
    test_err, test_loss = epoch(mnist_test_dataloader, model, None)
    return train_err, train_loss, test_err, test_loss
    ### END YOUR SOLUTION


if __name__ == "__main__":
    path = "/home/zhuyangyang/Course/CMU10_414/homework/hw2/data/"
    train_mnist(data_dir=path)
