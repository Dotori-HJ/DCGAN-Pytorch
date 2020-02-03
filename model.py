import math
import torch.nn as nn

from config import get_config


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.cfg = get_config()

        nf = self.cfg['MODEL']['N_FEATURES']
        nz = self.cfg['MODEL']['Z_DIM']
        nc = self.cfg['DATASET']['N_CHANNELS']
        im_size = self.cfg['DATASET']['IM_SIZE']
        nb = int(math.log2(im_size) - 3)

        layers = [
            nn.ConvTranspose2d(nz, nf * 2 ** nb, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(nf * 2 ** nb),
            nn.ReLU(inplace=True),
        ]
        for i in range(nb):
            layers.append(nn.ConvTranspose2d(nf * 2 ** (nb - i), nf * 2 ** (nb - i - 1), kernel_size=4, stride=2, padding=1, bias=False))
            layers.append(nn.BatchNorm2d(nf * 2 ** (nb - i - 1)))
            layers.append(nn.ReLU(inplace=True))

        layers.append(nn.ConvTranspose2d(nf, nc, kernel_size=4, stride=2, padding=1, bias=True))
        layers.append(nn.Tanh())

        self.main = nn.Sequential(*layers)

    def forward(self, input):
        return self.main(input)


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.cfg = get_config()

        nf = self.cfg['MODEL']['N_FEATURES']
        nc = self.cfg['DATASET']['N_CHANNELS']
        im_size = self.cfg['DATASET']['IM_SIZE']
        nb = int(math.log2(im_size) - 3)

        layers = [
            nn.Conv2d(nc, nf, kernel_size=4, stride=2, padding=1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
        ]
        for i in range(nb, 0, -1):
            layers.append(nn.Conv2d(nf * 2 ** (nb - i), nf * 2 ** (nb - i + 1), kernel_size=4, stride=2, padding=1, bias=False))
            layers.append(nn.BatchNorm2d(nf * 2 ** (nb - i + 1)))
            layers.append(nn.LeakyReLU(0.2, inplace=True))

        layers.append(nn.Conv2d(nf * 2 ** nb, 1, kernel_size=4, stride=1, padding=0, bias=True))
        layers.append(nn.Sigmoid())

        self.main = nn.Sequential(*layers)

    def forward(self, input):
        return self.main(input)


if __name__ == '__main__':
    from config import set_config
    set_config('configs/mnist.yaml')
    G = Generator()
    print(G)
    D = Discriminator()
    print(D)