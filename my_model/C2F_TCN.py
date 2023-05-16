import torch.nn.functional as F
import math
import torch.nn as nn
import torch
from functools import partial
import torchvision.models as mdels

nonlinearity = partial(F.relu, inplace=True)

class double_conv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size=5, padding=2),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_ch, out_ch, kernel_size=5, padding=2),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class inconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(inconv, self).__init__()
        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x):
        x = self.conv(x)
        return x

class outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size=1)

    def forward(self, x):
        x = self.conv(x)
        return x

class down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(down, self).__init__()
        self.max_pool_conv = nn.Sequential(
            nn.MaxPool1d(2), double_conv(in_ch, out_ch))

    def forward(self, x):
        x = self.max_pool_conv(x)
        return x

class up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        if bilinear:
            self.up = nn.Upsample(
                scale_factor=2, mode="linear", align_corners=True)
        else:
            self.up = nn.ConvTranspose1d(
                in_channels // 2, in_channels // 2, kernel_size=2, stride=2
            )

        self.conv = double_conv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diff = torch.tensor([x2.size()[2] - x1.size()[2]])

        x1 = F.pad(x1, [diff // 2, diff - diff //2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class TPPblock(nn.Module):
    def __init__(self, in_channels):
        super(TPPblock, self).__init__()
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.pool2 = nn.MaxPool1d(kernel_size=3, stride=3)
        self.pool3 = nn.MaxPool1d(kernel_size=5, stride=5)
        self.pool4 = nn.MaxPool1d(kernel_size=6, stride=6)

        self.conv = nn.Conv1d(
            in_channels=in_channels, out_channels=1, kernel_size=1, padding=0
        )

    def forward(self, x):
        self.in_channels, t = x.size(1), x.size(2)
        self.layer1 = F.interpolate(
            self.conv(self.pool1(x)), size=t, mode="linear", align_corners=True
        )
        self.layer2 = F.interpolate(
            self.conv(self.pool2(x)), size=t, mode="linear", align_corners=True
        )
        self.layer3 = F.interpolate(
            self.conv(self.pool3(x)), size=t, mode="linear", align_corners=True
        )
        self.layer4 = F.interpolate(
            self.conv(self.pool4(x)), size=t, mode="linear", align_corners=True
        )

        out = torch.cat([self.layer1, self.layer2,
                         self.layer3, self.layer4, x], 1)

        return out


class C2F_TCN(nn.Module):
    '''
        Features are extracted at the last layer of decoder.
    '''
    def __init__(self, n_channels, n_classes):
        super(C2F_TCN, self).__init__()
        self.inc = inconv(n_channels, 256)
        self.down1 = down(256, 256)
        self.down2 = down(256, 256)
        self.down3 = down(256, 128)
        self.down4 = down(128, 128)
        self.down5 = down(128, 128)
        self.down6 = down(128, 128)
        self.up = up(260, 128)
        self.outcc0 = outconv(128, n_classes)
        self.up0 = up(256, 128)
        self.outcc1 = outconv(128, n_classes)
        self.up1 = up(256, 128)
        self.outcc2 = outconv(128, n_classes)
        self.up2 = up(384, 128)
        self.outcc3 = outconv(128, n_classes)
        self.up3 = up(384, 128)
        self.outcc4 = outconv(128, n_classes)
        self.up4 = up(384, 128)
        self.outcc = outconv(128, n_classes)
        self.tpp = TPPblock(128)
        self.weights = torch.nn.Parameter(torch.ones(6))

        self.boundary_fc0 = nn.Conv1d(128, 64, 1, stride=1, padding=0)
        self.boundary_fc = nn.Conv1d(64, 2, 1, stride=1, padding=0)

    def forward(self, x):
        x1 = self.inc(x.squeeze(-1))
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x6 = self.down5(x5)
        x7 = self.down6(x6)
        # x7 = self.dac(x7)
        x7 = self.tpp(x7)
        x = self.up(x7, x6)
        y1 = self.outcc0(F.relu(x))
        # print("y1.shape=", y1.shape)
        x = self.up0(x, x5)
        y2 = self.outcc1(F.relu(x))
        # print("y2.shape=", y2.shape)
        x = self.up1(x, x4)
        y3 = self.outcc2(F.relu(x))
        # print("y3.shape=", y3.shape)
        x = self.up2(x, x3)
        y4 = self.outcc3(F.relu(x))
        # print("y4.shape=", y4.shape)
        x = self.up3(x, x2)
        y5 = self.outcc4(F.relu(x))
        # print("y5.shape=", y5.shape)
        x = self.up4(x, x1)
        y = self.outcc(x)
        boundary = self.boundary_fc0(x)
        boundary = self.boundary_fc(boundary)

        # print("y.shape=", y.shape)
        # return y, [y5, y4, y3, y2, y1], x

        return y,boundary




if __name__ == '__main__':

    def get_c2f_ensemble_output(outp, weights):
        ensemble_prob = F.softmax(outp[0], dim=1) * weights[0] / sum(weights)

        for i, outp_ele in enumerate(outp[1]):
            upped_logit = F.interpolate(outp_ele, size=outp[0].shape[-1], mode='linear', align_corners=True)
            ensemble_prob = ensemble_prob + F.softmax(upped_logit, dim=1) * weights[i + 1] / sum(weights)

        return ensemble_prob


    ensem_weights = [1, 1, 1, 1, 0, 0]

    model = C2F_TCN(16,11).cuda()

    x = torch.rand((4, 16, 900)).cuda()
    outputs_list = model(x)
    # outputs_ensemble = get_c2f_ensemble_output(outputs_list, ensem_weights)
    # print(outputs_ensemble.shape)
    print(outputs_list.shape)


