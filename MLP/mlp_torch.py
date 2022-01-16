import torch
from torch import nn
from torch.nn import init
import numpy as np
import sys
sys.path.append("..")
import d2lzh_pytorch as d2l


if __name__ == '__main__':
    num_inputs, num_outputs, num_hiddens = 784, 10, 256

    net = nn.Sequential(
        d2l.FlattenLayer(),
        nn.Linear(num_inputs, num_hiddens),
        nn.ReLU(),
        nn.Linear(num_hiddens, num_outputs)
    )

    for param in net.parameters():
        init.normal_(param, mean=0, std=0.01)

    batch, lr, num_epoch = 256, 0.01, 30
    loss = nn.CrossEntropyLoss()
    optim = torch.optim.SGD(net.parameters(), lr=lr)

    train_iter, test_iter = d2l.load_data_fashion_mnist(batch)

    for epoch in range(num_epoch):
        for X, y in train_iter:
            y_hat = net(X)
            l = loss(y_hat, y).sum()

            optim.zero_grad()

            l.backward()
            optim.step()

            train_loss = l.mean().item()
            train_acc = (y_hat.argmax(dim=1) == y).float().mean().item()

        test_acc = d2l.evaluate_accuracy(test_iter, net)
        print("epoch: {}, test acc: {}".format(epoch, test_acc))
