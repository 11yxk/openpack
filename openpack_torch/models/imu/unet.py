from typing import List

import torch
import torch.nn.functional as F
from torch import nn


class DownBlock(nn.Module):
    """A single down-sampling operation for U-Net's encoder.
    Attributes:
        double_conv (nn.Module): -
        pool (nn.MaxPool2d): -
    """

    def __init__(
            self,
            in_ch: int,
            out_ch: int,
            kernel_size: int = 3,
            pool_size: int = 2,
    ):
        """
        Args:
            in_ch/out_ch (int): input/output channels.
            kernel_size (int): kernel size for convolutions.
            pool_size (int): kernel size of a pooling.
        """
        super().__init__()

        self.double_conv = nn.Sequential(
            nn.Conv2d(
                in_ch,
                out_ch,
                kernel_size=(kernel_size, 1),
                stride=(1, 1),
                padding=(kernel_size // 2, 0),
            ),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
            nn.Conv2d(
                out_ch,
                out_ch,
                kernel_size=(kernel_size, 1),
                stride=(1, 1),
                padding=(kernel_size // 2, 0),
            ),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
        )
        self.pool = nn.MaxPool2d(kernel_size=(pool_size, 1))

    def forward(self, x: torch.Tensor):
        """
        Args:
            x (torch.Tensor)
        Returns:
             x, x_xskip:
                 * x (torch.Tensor): encoded tensor.
                 * x_skip (torch.Tensor): tensor to make a skip connection.
        """
        x_skip = self.double_conv(x)
        x = self.pool(x_skip)
        return x, x_skip


class UpBlock(nn.Module):
    """A single upsampling operation for U-Net's encoder.
    Attributes:
        up (nn.Upsampling or nn.ConvTransposed2d): -
        double_conv (DoubleConvBlock): -
    Note:
        ``padding`` is allways set to 'same'.
    """

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3):
        """
        Args:
            in_ch (int):
                the number of input channels of ``x1`` (main stream).
            out_ch (int): output channels. Usually, set ``in_ch // 2``.
            pool_size (int): kernel_size for corresponding pooling operation.
        Note:
            ``x2`` (skip connection) should have ``in_ch//`` channels.
        """
        super().__init__()
        # -- Upsamplomg Layer --
        # NOTE: Bilinear Inerpolation with Conv is better than ConvTranspose2d?
        self.up = nn.ConvTranspose2d(
            in_ch, out_ch, (1, 3), stride=(1, 2), padding=(0, 1)
        )
        # --  Double Conv Layer --
        self.double_conv = nn.Sequential(
            nn.Conv2d(
                out_ch * 2,
                out_ch,
                kernel_size=(kernel_size, 1),
                stride=(1, 1),
                padding=(kernel_size // 2, 0),
            ),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
            nn.Conv2d(
                out_ch,
                out_ch,
                kernel_size=(kernel_size, 1),
                stride=(1, 1),
                padding=(kernel_size // 2, 0),
            ),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
        )

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x1 (torch.Tensor): a tensor from main stream. shape = (N, C, H(=T), W)
            x2 (torch.Tensor): a skip connection tensor from downsampling layer.
                The shape should be (N, C//2, T*2, W).
        Returns:
            torch.Tensor
        """
        assert x1.size(1) == x2.size(1) * 2, f"x1={x1.size()}, x2={x2.size()}"
        assert abs(x1.size(2) - x2.size(2) //
                   2) < 3, f"x1={x1.size()}, x2={x2.size()}"

        # -- upsampling --
        x1 = self.up(x1)

        # -- Concat --
        diff_h = x2.size()[2] - x1.size()[2]
        diff_w = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diff_w // 2, diff_w - diff_w //
                        2, diff_h // 2, diff_h - diff_h // 2])
        x = torch.cat([x1, x2], dim=1)

        # -- conv --
        x = torch.cat([x1, x2], dim=1)
        x = self.double_conv(x)
        return x


# -----------------------------------------------------------------------------

class UNetEncoder(nn.ModuleList):
    """
    Attributes:
        depth (int):
            the number of ``DownBlock``.
        pools ([int]):
            list of kernel sizes for pooling.
        conv_blocks (nn.ModuleList): list of ``DownBlock``.
    Todo:
        implement ``get_output_ch(block_index)`` and remove ``filters``.
    """

    def __init__(self, ch_enc: int = 32, depth: int = 5, kernel_size: int = 3):
        super().__init__()
        self.depth = depth

        # -- main blocks --
        input_channels = tuple(
            [ch_enc] + [ch_enc * (2 ** i) for i in range(self.depth - 1)]
        )  # list of input channels.

        blocks = []
        for i, in_ch in enumerate(input_channels):
            if i == 0:
                blocks.append(DownBlock(in_ch, in_ch, pool_size=2))
            else:
                blocks.append(DownBlock(in_ch, in_ch * 2, pool_size=2))
        self.conv_blocks = nn.ModuleList(blocks)

        # -- bottom --
        in_ch = input_channels[-1] * 2
        self.bottom = nn.Sequential(
            nn.Conv2d(
                in_ch,
                in_ch * 2,
                kernel_size=(kernel_size, 1),
                stride=(1, 1),
                padding=(kernel_size // 2, 0),
            ),
            nn.BatchNorm2d(in_ch * 2),
            nn.ReLU(),
            nn.Conv2d(
                in_ch * 2,
                in_ch * 2,
                kernel_size=(kernel_size, 1),
                stride=(1, 1),
                padding=(kernel_size // 2, 0),
            ),
            nn.BatchNorm2d(in_ch * 2),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): shape=(B,C,T,W)
        Returns:
             encoded, skip_connections
                  * encoded (torch.Tensor): -
                  * skip_connections (list of torch.Tensor): -
        """
        # -- donwnsampling blocks --
        skip_connections = []
        for i in range(self.depth):
            x, x_skip = self.conv_blocks[i](x)
            skip_connections.append(x_skip)

        # -- bottom --
        encoded = self.bottom(x)

        return encoded, skip_connections


class UNetDecoder(nn.ModuleList):
    """
    Attributes:
        depth (int):
            the number of ``DownBlock``.
        up_blocks (nn.ModuleList):
            list of ``DownBlock``.
    """

    def __init__(self, ch_enc: int = 32, depth=5):
        """
        Args:
            ch_enc (int): the output channels of the 1st conv block.
            pools ([int]):
                list of kernel sizes for pooling.
        """
        super().__init__()
        self.depth = depth

        # -- main blocks --
        output_channels = tuple(
            reversed([ch_enc * (2 ** i) for i in range(self.depth)])
        )  # list of output channels.

        blocks = []
        for in_ch in output_channels:
            blocks.append(UpBlock(in_ch * 2, in_ch))
        self.up_blocks = nn.ModuleList(blocks)

    def forward(self, x: torch.Tensor,
                x_skips: List[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            x (Tensor): input
            x_skips ([Tensor]): Output of UTimeEncoder.
        """
        for i in range(self.depth):
            i_inv = (self.depth - 1) - i
            x = self.up_blocks[i](x, x_skips[i_inv])
        return x

# -----------------------------------------------------------------------------


class UNet(nn.Module):
    """
    Input must take channel-first format (BCHW).
    This model use 2D convolutional filter with kernel size = (f x 1).
    See also original U-net paper at http://arxiv.org/abs/1505.04597
    Note:
        Time axis should come in the 3rd dimention (i.e., H).
    """

    def __init__(
        self,
        in_ch: int = 6,
        num_classes: int = None,
        ch_inc: int = 32,
        depth: int = 5,
    ):
        """
        Args:
            in_ch (int): -
            num_classes (int): The number of classes to model.
            ch_inc (int, optional):
                the number of input channels for UNetEncoder. (Default: 32)
            pools (tuple of int):
               list of kernel sizes for pooling operations.
            depth (int): the number of blocks for Encoder/Decoder.
        """
        super().__init__()

        # NOTE: Add input encoding layer (UNet)
        # Ref:
        # https://github.com/milesial/Pytorch-UNet/blob/master/unet/unet_model.py
        self.inc = nn.Sequential(
            nn.Conv2d(
                in_ch,
                ch_inc,
                kernel_size=(3, 1),
                stride=(1, 1),
                padding=(1, 0),
            ),
            nn.BatchNorm2d(ch_inc),
            nn.ReLU(),
        )
        self.encoder = UNetEncoder(ch_inc, depth=depth)
        self.decoder = UNetDecoder(ch_inc, depth=depth)
        self.dense_clf = nn.Conv2d(ch_inc, num_classes, 1, padding=0, stride=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.inc(x)
        (x, res) = self.encoder(x)
        x = self.decoder(x, res)
        x = self.dense_clf(x)
        return x
