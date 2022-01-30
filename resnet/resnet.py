import torch
from torch import nn
from torch.nn import functional as F
import torchvision
import time


class Residual(nn.Module):
    def __init__(self, input_channel, num_channel,
                 use_1x1conv=False, stride=1):

        super().__init__()

        self.conv1 = nn.Conv2d(in_channels=input_channel, out_channels=num_channel,
                               kernel_size=3, padding=1,
                               stride=stride)  # use stride to down sampling, this stride correspond to conv3's stride

        self.conv2 = nn.Conv2d(in_channels=num_channel, out_channels=num_channel,
                               kernel_size=3, padding=1)

        if use_1x1conv:
            self.conv3 = nn.Conv2d(in_channels=input_channel, out_channels=num_channel,
                                   kernel_size=1, stride=stride)
        else:
            self.conv3 = None

        self.bn1 = nn.BatchNorm2d(num_channel)
        self.bn2 = nn.BatchNorm2d(num_channel)

    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        Y += X
        return F.relu(Y)


def resnet_block(input_channel, num_channel, num_residual, first_block=False):
    blk = []

    for i in range(num_residual):
        if i == 0 and not first_block:
            blk.append(Residual(input_channel, num_channel, use_1x1conv=True, stride=2))
        else:
            blk.append(Residual(num_channel, num_channel))
    return blk


def load_data_fashion_mnist(batch_size, resize=None, root='../dataset/FashionMNIST'):
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


def sgd(params, lr, batch_size):
    for param in params:
        #  Modifying param with param.data will not be passed to the calculation diagram
        param.data -= lr * param.grad / batch_size


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

    if loss is None:
        loss = torch.nn.CrossEntropyLoss()

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


if __name__ == "__main__":
    b1 = nn.Sequential(nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
                       nn.BatchNorm2d(64), nn.ReLU(),
                       nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

    b2 = nn.Sequential(*resnet_block(64, 64, num_residual=2, first_block=True))  # without 1x1
    b3 = nn.Sequential(*resnet_block(64, 128, num_residual=2))
    b4 = nn.Sequential(*resnet_block(128, 256, num_residual=2))
    b5 = nn.Sequential(*resnet_block(256, 512, num_residual=2))

    net = nn.Sequential(b1, b2, b3, b4, b5,
                        nn.AdaptiveAvgPool2d((1, 1)),
                        nn.Flatten(), nn.Linear(512, 10))

    lr, num_epochs, batch_size = 0.05, 10, 64
    train_iter, test_iter = load_data_fashion_mnist(batch_size, resize=95)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    train_func(net, train_iter, test_iter, None, num_epochs, batch_size, device, lr=lr, optimizer=optimizer)
