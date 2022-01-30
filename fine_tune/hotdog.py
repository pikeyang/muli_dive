import torch

from torchvision import transforms, models
from torchvision.datasets import ImageFolder
from torch import nn, optim
from torch.utils.data import DataLoader

import sys
sys.path.append('..')
import d2lzh_pytorch as d2l


def train_fine_tuning(net, train_imgs, test_imgs, batch_size=128, num_epoch=5, device=None, optimizer = optim.SGD):
    train_iter = DataLoader(train_imgs, batch_size=batch_size, shuffle=True)
    test_iter = DataLoader(test_imgs, batch_size=batch_size, shuffle=False)

    loss = nn.CrossEntropyLoss()

    if device is None:
        device = torch.device('cpu')

    d2l.train_func(net, train_iter, test_iter, loss, num_epoch, batch_size=batch_size, device=device, optimizer=optimizer)


if __name__ == '__main__':

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    train_augs = transforms.Compose([
        transforms.RandomResizedCrop(size=224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ])

    test_augs = transforms.Compose([
        transforms.Resize(size=256),
        transforms.CenterCrop(size=224),
        transforms.ToTensor(),
        normalize
    ])

    train_img = ImageFolder('../dataset/hotdog/train', transform=train_augs)
    test_img = ImageFolder('../dataset/hotdog/test', transform=test_augs)

    resnet = models.resnet18(pretrained=True)
    resnet.fc = nn.Linear(512, 2)

    # id of full connection layer
    output_params = list(map(id, resnet.fc.parameters()))
    feature_params = filter(lambda p: id(p) not in output_params, resnet.parameters())

    lr = 0.01
    optimizer = optim.SGD([{'params': feature_params},
                           {'params': resnet.fc.parameters(), 'lr': lr*10}],
                          lr=lr, weight_decay=0.001)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    train_fine_tuning(resnet, train_img, test_img, device=device, optimizer=optimizer)
