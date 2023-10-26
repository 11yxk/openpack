""" Implementation of DeepConvLSTM
Reference:
- https://www.mdpi.com/1424-8220/16/1/115
"""
import torch
from torch import nn


class DeepConvLSTM(nn.Module):
    """Imprementation of DeepConvLSTM [Sensors 2016].

    Note:
        https://www.mdpi.com/1424-8220/16/1/115 (Sensors, 2016)

    """

    def __init__(self, in_ch: int = 6, num_classes: int = None):
        super().__init__()

        # NOTE: The first block is input layer.

        # -- [L2-5] Convolutions --
        blocks = []
        for i in range(4):
            in_ch_ = in_ch if i == 0 else 64
            blocks.append(
                nn.Sequential(
                    nn.Conv2d(in_ch_, 64, kernel_size=(5, 1), padding=(2, 0)),
                    nn.BatchNorm2d(64),
                    nn.ReLU(),
                )
            )
        self.conv2to5 = nn.ModuleList(blocks)

        # -- [L6-7] LSTM --
        hidden_units = 128
        self.lstm6 = nn.LSTM(64, hidden_units, batch_first=True)
        self.lstm7 = nn.LSTM(hidden_units, hidden_units, batch_first=True)
        self.dropout6 = nn.Dropout(p=0.5)
        self.dropout7 = nn.Dropout(p=0.5)

        # -- [L8] Softmax Layer (Output Layer) --
        self.out8 = nn.Conv2d(
            hidden_units,
            num_classes,
            1,
            stride=1,
            padding=0,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): shape = (B, CH, T, 1)
        Returns:
            torch.Tensor: shape = (B, N_CLASSES, T, 1)
        """
        # -- Conv --
        for i in range(4):
            x = self.conv2to5[i](x)

        # -- LSTM --
        # Reshape: (B, CH, 1, T) -> (B, T, CH)
        x = x.squeeze(3).transpose(1, 2)

        x, _ = self.lstm6(x)
        x = self.dropout6(x)
        x, _ = self.lstm7(x)
        x = self.dropout7(x)

        # -- [L8] Softmax Layer (Output Layer) --
        # Reshape: (B, T, CH) -> (B, CH, T, 1)
        x = x.transpose(1, 2).unsqueeze(3)
        x = self.out8(x)
        return x


class DeepConvLSTMSelfAttn(nn.Module):
    """Imprementation of a DeepConvLSTM with Self-Attention used in ''Deep ConvLSTM with
    self-attention for human activity decoding using wearable sensors'' (Sensors 2020).

    Note:
        https://ieeexplore.ieee.org/document/9296308 (Sensors 2020)
    """

    def __init__(
        self,
        in_ch: int = 6,
        num_classes: int = 11,
        cnn_filters=3,
        lstm_units=32,
        num_attn_heads: int = 1,
    ):
        super().__init__()

        # NOTE: The first block is input layer.

        # -- [1] Embedding Layer --
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, cnn_filters, kernel_size=1, padding=0),
            nn.BatchNorm2d(cnn_filters),
            nn.ReLU(),
        )

        # -- [2] LSTM Encoder --
        self.lstm = nn.LSTM(cnn_filters, lstm_units, batch_first=True)
        self.dropout = nn.Dropout(p=0.5)

        # -- [3] Self-Attention --
        self.attention = nn.MultiheadAttention(
            lstm_units,
            num_attn_heads,
            batch_first=True,
        )

        # -- [4] Softmax Layer (Output Layer) --

        self.out = nn.Conv2d(
            lstm_units,
            num_classes,
            kernel_size=(1,1),
            stride=(1,1),
            padding=0,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): shape = (B, CH, T, 1)
        Returns:
            torch.Tensor: shape = (B, N_CLASSES, T, 1)
        """
        # -- [1] Embedding Layer --
        x = self.conv(x)

        # -- [2] LSTM Encoder --
        # Reshape: (B, CH, 1, T) -> (B, T, CH)
        x = x.squeeze(3).transpose(1, 2)

        x, _ = self.lstm(x)
        x = self.dropout(x)

        # -- [3] Self-Attention --
        x, w = self.attention(x.clone(), x.clone(), x.clone())

        # -- [4] Softmax Layer (Output Layer) --
        # Reshape: (B, T, CH) -> (B, CH, T, 1)
        x = x.transpose(1, 2).unsqueeze(3)
        x = self.out(x)
        return x


if __name__ == '__main__':


    model=DeepConvLSTMSelfAttn().cuda()
    x=torch.randn((8,6,150,1)).cuda()
    op=model(x)
    print("Op shape: ",op.shape)


    def count_param(model):
        param_count = 0
        for param in model.parameters():
            param_count += param.view(-1).size()[0]
        print(param_count)
        return param_count
    count_param(model)


    from thop import profile


    flops, params = profile(model, inputs=(x,))
    total = sum([param.nelement() for param in model.parameters()])
    print('   Number of params: %.2fM' % (total / 1e6))
    print('   Number of FLOPs: %.2fGFLOPs' % (flops / 1e9))