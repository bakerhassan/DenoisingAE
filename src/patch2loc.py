import torchvision
from torch import nn
import torch
from torch.nn import functional as F

import torch.distributions as td


class patch2loc(nn.Module):
    def __init__(self, activations=False, position_conditional=False,positional_encoding = False):
        super(patch2loc, self).__init__()
        self.net = torchvision.models.resnet152(weights=torchvision.models.ResNet152_Weights.IMAGENET1K_V2)
        self.position_conditional = position_conditional
        if position_conditional:
            if positional_encoding:
                self.net._modules['fc'] = nn.Linear(2048, 2048-100)
            else:
                self.net._modules['fc'] = nn.Linear(2048, 2047)
            output_dim = 2
        else:
            self.net._modules['fc'] = nn.Linear(2048, 2048)
            output_dim = 2
        n_channels = 1
        self.bn = nn.BatchNorm2d(n_channels)  # Adjust the number of input channels (3 in this example)
        new_conv1 = nn.Conv2d(n_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.net.conv1 = new_conv1

        # self.remove_batch_norm(self.net)
        self.branch1 = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, output_dim)
        )

        self.branch2 = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, output_dim)
        )

        self.mean_penultimate = nn.Linear(32, 6)
        self.activations = activations

    def forward(self, x, *kargs, **kwargs):
        x = self.bn(x)
        x = F.relu(self.net(x))

        if self.position_conditional:
            positions = kargs[0]['position']
            x = torch.cat((x, positions[:, None].to(torch.float32)), dim=1)
        if self.activations:
            return x
        mu = self.branch1(x)
        logvar = self.branch2(x)
        std = (logvar.exp() + 1e-10).pow(0.5)
        q_z = td.multivariate_normal.MultivariateNormal(mu, torch.diag_embed(std) + 1e-9)

        return q_z
