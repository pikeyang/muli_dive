import torch
import torchvision
import torch.nn as nn
import time


def get_fashion_mnist_labels(labels):
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
                   'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [text_labels[int(i)] for i in labels]


def load_data_fashion_mnist(batch_size, resize=None, root='./dataset/FashionMNIST'):
    """Download the fashion mnist dataset and then load into memory."""
    trans = []
    if resize:
        trans.append(torchvision.transforms.Resize(size=resize))
    trans.append(torchvision.transforms.ToTensor())

    transform = torchvision.transforms.Compose(trans)
    mnist_train = torchvision.datasets.FashionMNIST(root=root, train=True,
                                                    download=True, transform=transform)
    mnist_test = torchvision.datasets.FashionMNIST(root=root, train=False,
                                                   download=True, transform=transform)

    train_iter = torch.utils.data.DataLoader(mnist_train, batch_size=batch_size, shuffle=True, num_workers=4)
    test_iter = torch.utils.data.DataLoader(mnist_test, batch_size=batch_size, shuffle=False, num_workers=4)

    return train_iter, test_iter


def sgd(params, lr, batch_size):
    for param in params:
        #  Modifying param with param.data will not be passed to the calculation diagram
        param.data -= lr * param.grad / batch_size


def evaluate_accuracy(data_iter, net, device=None):
    if device is None and isinstance(net, torch.nn.Module):
        device = list(net.parameters())[0].device

    acc_sum, n = 0.0, 0
    with torch.no_grad():
        for X, y in data_iter:
            if isinstance(net, torch.nn.Module):
                net.eval()
                acc_sum += (net(X.to(device)).argmax(dim=1) == y.to(device)).float().sum()
                net.train()
            else:  # 自定义的模型, 3.13节之后不会用到, 不考虑GPU
                if 'is_training' in net.__code__.co_varnames:  # 如果有is_training这个参数
                    # 将is_training设置成False
                    acc_sum += (net(X, is_training=False).argmax(dim=1) == y).float().sum().item()
                else:
                    acc_sum += (net(X).argmax(dim=1) == y).float().sum().item()
            n += y.shape[0]
    return acc_sum / n


def train_func(net: nn.Module, train_iter, test_iter, loss, num_epoch: int, batch_size: int, device,
               params=None, lr=None,
               optimizer=None,
               writer=None):
    """
    :param net:
    :param train_iter:
    :param test_iter:
    :param loss:
    :param num_epoch:
    :param batch_size:
    :param device:
    :param params:
    :param lr:
    :param optimizer:
    :param writer: for tensorboard
    :return:
    """
    net = net.to(device)
    print("training on ", device)

    for epoch in range(num_epoch):
        train_loss_sum, train_acc_sum, n, start = 0.0, 0.0, 0, time.time()
        for batch_count, (X, y) in enumerate(train_iter):
            X = X.to(device)
            y = y.to(device)
            y_hat = net(X)
            loss_fn = loss(y_hat, y)

            if optimizer is not None:  # use module
                optimizer.zero_grad()
            elif params is not None and params[0].grad is not None:
                for param in params:
                    param.grad.data.zero_()

            loss_fn.backward()

            if optimizer is None:
                sgd(params, lr, batch_size)
            else:
                optimizer.step()

            train_loss_sum += loss_fn.cpu().item()
            train_acc_sum += (y_hat.argmax(dim=1) == y).cpu().sum().item()
            n += y.shape[0]

            if writer is not None:
                if batch_count % 100 == 99:
                    writer.add_scalar('train_loss',
                                      train_loss_sum / n,
                                      epoch * len(train_iter) + batch_count)

        test_acc = evaluate_accuracy(test_iter, net)
        print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f, time %.1f sec'
              % (epoch + 1, train_loss_sum / n, train_acc_sum / n, test_acc, time.time() - start))


class FlattenLayer(nn.Module):
    def __init__(self):
        super(FlattenLayer, self).__init__()

    def forward(self, x):
        return x.view(x.shape[0], -1)


if __name__ == '__main__':
    pass
