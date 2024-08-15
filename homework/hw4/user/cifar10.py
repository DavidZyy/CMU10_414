import needle as ndl
from models import ResNet9
from simple_ml import train_cifar10, evaluate_cifar10


if __name__ == "__main__":
    # device = ndl.cpu()
    device = ndl.cuda()
    base_folder = "/home/zhuyangyang/Course/CMU10_414/homework/hw4/data/cifar-10-batches-py"
    dataset_train = ndl.data.CIFAR10Dataset(base_folder, train=True)
    dataset_test = ndl.data.CIFAR10Dataset(base_folder, train=False)
    batch_size = 128
    dataloader_train = ndl.data.DataLoader(dataset_train, batch_size, shuffle=True, device=device)
    dataloader_test = ndl.data.DataLoader(dataset_test, batch_size, shuffle=True, device=device)
    model = ResNet9(device=device, dtype="float32")
    evaluate_cifar10(model, dataloader_test)  # loss maybe inf, but after one epoch train, it will down.
    train_cifar10(model, dataloader_train, n_epochs=1)
    evaluate_cifar10(model, dataloader_test)
