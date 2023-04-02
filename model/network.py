import torch as tc
import torch.nn as nn
import torch.nn.functional as F
from model.resnet34 import ResNet34
from model.resnet50 import ResNet50

class LeNet5(nn.Module):
    def __init__(self, num_classes=100):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)  # 3*16*25=1200
        self.pool1 = nn.MaxPool2d(2, 2)  #
        self.conv2 = nn.Conv2d(16, 32, 5)  # 16*32*25=12800
        self.pool2 = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(32 * 4 * 4, 256)  # 32*4*4*128=65536
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))  # input(3,32,32) output(16,28,28)
        x = self.pool1(x)  # output(16，14，14)
        x = F.relu(self.conv2(x))  # output(32,10.10)
        x = self.pool2(x)  # output(32,5,5)
        x = F.relu(self.fc1(x))  # output(120)
        x = F.relu(self.fc2(x))  # output(84)
        x = self.fc3(x)  # output(10)
        return x


class FC2(nn.Module):
    def __init__(self, num_classes=100):
        super(FC2, self).__init__()
        self.conv = nn.Conv2d(3, 1, 1)
        self.fc1 = nn.Linear(32 * 32 * 1, 256)  # 32*4*4*128=65536
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.conv(x)
        x = x.view(-1, 32 * 32 * 1)
        return self.fc2(F.relu(self.fc1(x)))


vgg = [96, 96, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, '512', 'M']


class VGG16(nn.Module):
    def __init__(self, num_classes=100):
        super(VGG16, self).__init__()
        self.features = self._make_layers(vgg)
        self.dense = nn.Sequential(
            nn.Linear(512, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
        )
        self.classifier = nn.Linear(4096, num_classes)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.dense(out)
        out = self.classifier(out)
        return out

    def _make_layers(self, vggs):
        layers = []
        in_channels = 3
        index = 0
        for x in vggs:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
                index += 1
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=(3, 3), padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
                index += 3
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)


alexnet = [64, 'M', 192, 'M', 384, 256, 256, 'M']
class AlexNet(nn.Module):

    def __init__(self, num_classes=100):
        super(AlexNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1)
        self.relu1 = nn.ReLU(inplace=True)
        self.features = self.make_layers(alexnet)
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 1 * 1, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def make_layers(self, alexnets):
        layers = []
        in_channels = 64
        index = 0
        for x in alexnets:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=3, stride=2)]
                index += 1
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.ReLU(inplace=True)]
                in_channels = x
                index += 2
        return nn.Sequential(*layers)

    def forward(self, x):

        x = self.relu1(self.conv1(x))
        x = self.features(x)
        x = tc.flatten(x, 1)
        x = self.classifier(x)
        return x


def choose_network(model, num_classes=100):
    if model == 'FC2':
        nn = FC2(num_classes=num_classes)
    elif model == 'lenet5':
        nn = LeNet5(num_classes=num_classes)
    elif model == 'vgg16':
        nn = VGG16(num_classes)
    elif model == 'alexnet':
        nn = AlexNet(num_classes)
    elif model == 'ResNet34':
        nn = ResNet34(num_classes=num_classes)
    elif model == 'ResNet50':
        nn = ResNet50(num_classes=num_classes)
    return nn
