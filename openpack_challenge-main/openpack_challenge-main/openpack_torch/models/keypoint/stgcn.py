"""Ref: https://colab.research.google.com/github/machine-perception-robotics-group/MPRGDeepLearningLectureNotebook/blob/master/15_gcn/03_action_recognition_ST_GCN.ipynb#scrollTo=Vk-AMCVb5jqM
"""
import numpy as np
import torch
import torch.nn as nn


class SpatialGraphConvLayer(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, Ks: int):
        """Implementation of Spacial Graph Convolution Layer.
        Args:
            in_channels (int): _description_
            out_channels (int): _description_
            Ks (int): _description_
        """
        super().__init__()
        self.Ks = Ks
        self.conv = nn.Conv2d(in_channels=in_channels,
                              out_channels=out_channels * Ks,
                              kernel_size=1)

    def forward(self, x: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): shape=(N, CH, FRAMES, VERTEX)
            A (torch.Tensor): shape=(Ks, VERTEX, VERTEX)
        Returns:
            torch.Tensor: the same shape as input ``x``.
        """
        x = self.conv(x)
        n, kc, t, v = x.size()
        x = x.view(n, self.Ks, kc // self.Ks, t, v)

        # Apply GraphConv and sum up features.
        x = torch.einsum('nkctv,kvw->nctw', (x, A))
        return x.contiguous()


class TemporalConvLayer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        Kt: int,
        stride: int = 1,
        dropout: float = 0.5,
    ) -> None:
        """Implementation of temporal convolution layer.
        Args:
            in_channels (int): _description_
            Kt (int): kernel size for temporal domain.
            stride (int): stride for temporal domain.
            dropout (float): _description_
        """
        super().__init__()
        self.block = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv2d(in_channels,
                      in_channels,
                      (Kt, 1),
                      (stride, 1),
                      ((Kt - 1) // 2, 0)),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): shape=(BATCH, CH, FRAMES, VERTEX)
        Returns:
            torch.Tensor: the same shape as input
        """
        x = self.block(x)
        return x


class STConvBlock(nn.Module):
    """Implementation of Spatial-temporal convolutional block with
    learnable edge.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        Ks: int = None,
        Kt: int = None,
        num_vertex: int = None,
        stride: int = 1,
        dropout=0.5,
    ):
        super().__init__()
        # 空間グラフの畳み込み
        self.sgc = SpatialGraphConvLayer(
            in_channels=in_channels,
            out_channels=out_channels,
            Ks=Ks,
        )

        # Learnable weight matrix M エッジに重みを与えます. どのエッジが重要かを学習します.
        self.M = nn.Parameter(torch.ones((Ks, num_vertex, num_vertex)))
        self.tgc = TemporalConvLayer(out_channels, Kt, stride)

    def forward(self, x, A):
        x = self.sgc(x, A * self.M)
        x = self.tgc(x)
        return x

# -----------------------------------------------------------------------------


class STGCN4Seg(nn.Module):
    """Implementation of ST-GCN for segmentation task.

    """

    def __init__(
            self,
            in_channels: int = None,
            num_classes: int = None,
            Ks: int = None,
            Kt: int = None,
            A: np.ndarray = None,
    ):
        super().__init__()
        A = torch.tensor(A, dtype=torch.float32, requires_grad=False)
        self.register_buffer('A', A)
        A_size = A.size()
        num_vertex = A.size(1)

        # Batch Normalization
        self.bn = nn.BatchNorm1d(in_channels * A_size[1])

        # STConvBlocks
        self.stgc1 = STConvBlock(
            in_channels,
            32,
            Ks=Ks,
            Kt=Kt,
            num_vertex=num_vertex)
        self.stgc2 = STConvBlock(32, 32, Ks=Ks, Kt=Kt, num_vertex=num_vertex)
        self.stgc3 = STConvBlock(32, 32, Ks=Ks, Kt=Kt, num_vertex=num_vertex)
        self.stgc4 = STConvBlock(32, 64, Ks=Ks, Kt=Kt, num_vertex=num_vertex)

        self.stgc5 = STConvBlock(64, 64, Ks=Ks, Kt=Kt, num_vertex=num_vertex)
        self.stgc6 = STConvBlock(64, 64, Ks=Ks, Kt=Kt, num_vertex=num_vertex)
        self.stgc7 = STConvBlock(64, 64, Ks=Ks, Kt=Kt, num_vertex=num_vertex)
        self.stgc8 = STConvBlock(64, 64, Ks=Ks, Kt=Kt, num_vertex=num_vertex)

        self.stgc9 = STConvBlock(64, 64, Ks=Ks, Kt=Kt, num_vertex=num_vertex)
        self.stgc10 = STConvBlock(64, 64, Ks=Ks, Kt=Kt, num_vertex=num_vertex)
        self.stgc11 = STConvBlock(64, 64, Ks=Ks, Kt=Kt, num_vertex=num_vertex)
        self.stgc12 = STConvBlock(64, 64, Ks=Ks, Kt=Kt, num_vertex=num_vertex)

        # Prediction
        self.fc = nn.Conv2d(64, num_classes, kernel_size=(1, num_vertex))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): shape=(BATCH, IN_CH, FRAMES, VERTEX)
        Returns:
            torch.Tensor: the same shape as the input ``x``.
        """
        # Batch Normalization
        N, C, T, V = x.size()  # batch, channel, frame, node
        x = x.permute(0, 3, 1, 2).contiguous().view(N, V * C, T)
        x = self.bn(x)
        x = x.view(N, V, C, T).permute(0, 2, 3, 1).contiguous()

        # STGC_blocks
        x = self.stgc1(x, self.A)
        x = self.stgc2(x, self.A)
        x = self.stgc3(x, self.A)
        x = self.stgc4(x, self.A)

        x = self.stgc5(x, self.A)
        x = self.stgc6(x, self.A)
        x = self.stgc7(x, self.A)
        x = self.stgc8(x, self.A)

        x = self.stgc9(x, self.A)
        x = self.stgc10(x, self.A)
        x = self.stgc11(x, self.A)
        x = self.stgc12(x, self.A)

        # Prediction
        x = self.fc(x)
        return x
