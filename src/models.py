import torch.nn as nn
import torch.nn.init as init


class ResidualBlock(nn.Module):
    expansion = 4

    def __init__(self, channel1, channel2, stride=1, down_sample=None, dropout_prob=0.2):
        super(ResidualBlock, self).__init__()
        self.stride = stride
        self.down_sample = down_sample
        self.conv1 = nn.Conv2d(channel1, channel2, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channel2)
        self.conv2 = nn.Conv2d(channel2, channel2, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channel2)
        self.conv3 = nn.Conv2d(channel2, channel2 * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(channel2 * self.expansion)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout2d(p=dropout_prob)

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out = self.dropout(out)
        if self.down_sample is not None:
            residual = self.down_sample(x)
        out += residual
        out = self.relu(out)
        return out


class ResNet34(nn.Module):
    def __init__(self, block, num_classes=2):
        super(ResNet34, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self.layer_creator(block, 64, layers=3, stride=1)
        self.layer2 = self.layer_creator(block, 128, layers=4, stride=2)
        self.layer3 = self.layer_creator(block, 256, layers=6, stride=2)
        self.layer4 = self.layer_creator(block, 512, layers=3, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, 512)
        self.fc2 = nn.Linear(512, num_classes)

        # Initialize parameters
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)

    def layer_creator(self, block, channels, layers, stride=1):
        down_sample = None
        if stride != 1 or self.in_channels != channels * block.expansion:
            down_sample = nn.Sequential(
                nn.Conv2d(self.in_channels, channels * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(channels * block.expansion)
            )
        layers_list = [block(self.in_channels, channels, stride, down_sample)]
        self.in_channels = channels * block.expansion
        for _ in range(1, layers):
            layers_list.append(block(self.in_channels, channels))
        return nn.Sequential(*layers_list)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc(x))
        x = self.fc2(x)
        return x


def model_creator(num_classes):
    return ResNet34(ResidualBlock, num_classes)
