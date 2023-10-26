import math
import pdb

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from logging import getLogger
logger = getLogger(__name__)
def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod


def conv_branch_init(conv, branches):
    weight = conv.weight
    n = weight.size(0)
    k1 = weight.size(1)
    k2 = weight.size(2)
    nn.init.normal_(weight, 0, math.sqrt(2. / (n * k1 * k2 * branches)))
    nn.init.constant_(conv.bias, 0)


def conv_init(conv):
    if conv.weight is not None:
        nn.init.kaiming_normal_(conv.weight, mode='fan_out')
    if conv.bias is not None:
        nn.init.constant_(conv.bias, 0)


def bn_init(bn, scale):
    nn.init.constant_(bn.weight, scale)
    nn.init.constant_(bn.bias, 0)


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        if hasattr(m, 'weight'):
            nn.init.kaiming_normal_(m.weight, mode='fan_out')
        if hasattr(m, 'bias') and m.bias is not None and isinstance(m.bias, torch.Tensor):
            nn.init.constant_(m.bias, 0)
    elif classname.find('BatchNorm') != -1:
        if hasattr(m, 'weight') and m.weight is not None:
            m.weight.data.normal_(1.0, 0.02)
        if hasattr(m, 'bias') and m.bias is not None:
            m.bias.data.fill_(0)


class TemporalConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1):
        super(TemporalConv, self).__init__()

        pad = (kernel_size + (kernel_size-1) * (dilation-1) - 1) // 2
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(kernel_size, 1),
            padding=(pad, 0),
            stride=(stride, 1),
            dilation=(dilation, 1))

        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class MultiScale_TemporalConv(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 dilations=[1,2,3,4],
                 residual=True,
                 residual_kernel_size=1):

        super().__init__()
        # assert out_channels % (len(dilations) + 2) == 0, '# out channels should be multiples of # branches'
        assert out_channels % (len(dilations) ) == 0, '# out channels should be multiples of # branches'
        # Multiple branches of temporal convolution
        # self.num_branches = len(dilations) + 2
        self.num_branches = len(dilations)
        branch_channels = out_channels // self.num_branches
        if type(kernel_size) == list:
            assert len(kernel_size) == len(dilations)
        else:
            kernel_size = [kernel_size]*len(dilations)
        # Temporal Convolution branches
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    branch_channels,
                    kernel_size=1,
                    padding=0),
                nn.BatchNorm2d(branch_channels),
                nn.ReLU(inplace=True),
                TemporalConv(
                    branch_channels,
                    branch_channels,
                    kernel_size=ks,
                    stride=stride,
                    dilation=dilation),
            )
            for ks, dilation in zip(kernel_size, dilations)
        ])

        # Additional Max & 1x1 branch
        # self.branches.append(nn.Sequential(
        #     nn.Conv2d(in_channels, branch_channels, kernel_size=1, padding=0),
        #     nn.BatchNorm2d(branch_channels),
        #     nn.ReLU(inplace=True),
        #     nn.MaxPool2d(kernel_size=(3,1), stride=(stride,1), padding=(1,0)),
        #     nn.BatchNorm2d(branch_channels)  # 为什么还要加bn
        # ))
        #
        # self.branches.append(nn.Sequential(
        #     nn.Conv2d(in_channels, branch_channels, kernel_size=1, padding=0, stride=(stride,1)),
        #     nn.BatchNorm2d(branch_channels)
        # ))

        # Residual connection
        if not residual:
            self.residual = lambda x: 0
        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x
        else:
            self.residual = TemporalConv(in_channels, out_channels, kernel_size=residual_kernel_size, stride=stride)

        # initialize
        self.apply(weights_init)

    def forward(self, x):
        # Input dim: (N,C,T,V)
        res = self.residual(x)
        branch_outs = []
        for tempconv in self.branches:
            out = tempconv(x)
            branch_outs.append(out)


        out = torch.cat(branch_outs, dim=1)


        out += res
        return out


# class CTRGC(nn.Module):
#     def __init__(self, in_channels, out_channels, rel_reduction=8, mid_reduction=1):
#         super(CTRGC, self).__init__()
#         self.in_channels = in_channels
#         self.out_channels = out_channels
#         if in_channels == 3 or in_channels == 9:
#             self.rel_channels = 8
#             self.mid_channels = 16
#         else:
#             self.rel_channels = in_channels // rel_reduction
#             self.mid_channels = in_channels // mid_reduction
#         self.conv1 = nn.Conv2d(self.in_channels, self.rel_channels, kernel_size=1)
#         self.conv2 = nn.Conv2d(self.in_channels, self.rel_channels, kernel_size=1)
#         self.conv3 = nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1)
#         self.conv4 = nn.Conv2d(self.rel_channels, self.out_channels, kernel_size=1)
#         self.tanh = nn.Tanh()
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 conv_init(m)
#             elif isinstance(m, nn.BatchNorm2d):
#                 bn_init(m, 1)
#
#     def forward(self, x, A=None, alpha=1):
#         x1, x2, x3 = self.conv1(x).mean(-2), self.conv2(x).mean(-2), self.conv3(x)
#         x1 = self.tanh(x1.unsqueeze(-1) - x2.unsqueeze(-2))
#         x1 = self.conv4(x1) * alpha + (A.unsqueeze(0).unsqueeze(0) if A is not None else 0)  # N,C,V,V
#         x1 = torch.einsum('ncuv,nctv->nctu', x1, x3)
#         return x1

class CTRGC(nn.Module):
    def __init__(self, in_channels, out_channels, rel_reduction=8, mid_reduction=1):
        super(CTRGC, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        if in_channels == 3 or in_channels == 9:
            self.rel_channels = 8
            self.mid_channels = 16
        else:
            self.rel_channels = in_channels // rel_reduction
            self.mid_channels = in_channels // mid_reduction
        self.conv1 = nn.Conv2d(self.in_channels, self.rel_channels, kernel_size=1)
        self.conv2 = nn.Conv2d(self.in_channels, self.rel_channels, kernel_size=1)
        self.conv3 = nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1)
        self.conv4 = nn.Conv2d(self.rel_channels, self.out_channels, kernel_size=1)
        self.tanh = nn.Tanh()
        # self.soft = nn.Softmax(dim = -2)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)


    def forward(self, x, A=None, alpha=1,beta = 1):
        x1, x2, x3 = self.conv1(x), self.conv2(x), self.conv3(x)

        # temporal
        xt = self.tanh(x1.mean(1).unsqueeze(-1) - x2.mean(1).unsqueeze(-2)) # N,T,V,V
        xt = xt * beta + (A.unsqueeze(0).unsqueeze(0) if A is not None else 0)  # N,T,V,V
        xt = torch.einsum('ntuv,nctv->nctu', xt, x3)

        xc = self.tanh(x1.mean(-2).unsqueeze(-1) - x2.mean(-2).unsqueeze(-2))  # N,C,V,V
        xc = self.conv4(xc) * alpha + (A.unsqueeze(0).unsqueeze(0) if A is not None else 0)  # N,C,V,V
        xc = torch.einsum('ncuv,nctv->nctu', xc, x3)


        return xt+xc





class unit_tcn(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=9, stride=1):
        super(unit_tcn, self).__init__()
        pad = int((kernel_size - 1) / 2)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(kernel_size, 1), padding=(pad, 0),
                              stride=(stride, 1))

        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        conv_init(self.conv)
        bn_init(self.bn, 1)

    def forward(self, x):
        x = self.bn(self.conv(x))
        return x


class unit_gcn(nn.Module):
    def __init__(self, in_channels, out_channels, A, coff_embedding=4, adaptive=True, residual=True):
        super(unit_gcn, self).__init__()
        inter_channels = out_channels // coff_embedding
        self.inter_c = inter_channels
        self.out_c = out_channels
        self.in_c = in_channels
        self.adaptive = adaptive
        self.num_subset = A.shape[0]
        self.convs = nn.ModuleList()
        for i in range(self.num_subset):
            self.convs.append(CTRGC(in_channels, out_channels))

        if residual:
            if in_channels != out_channels:
                self.down = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, 1),
                    nn.BatchNorm2d(out_channels)
                )
            else:
                self.down = lambda x: x
        else:
            self.down = lambda x: 0
        if self.adaptive:
            self.PA = nn.Parameter(torch.from_numpy(A.astype(np.float32)))
        else:
            self.A = Variable(torch.from_numpy(A.astype(np.float32)), requires_grad=False)
        self.alpha = nn.Parameter(torch.zeros(1))
        self.beta = nn.Parameter(torch.zeros(1))
        self.bn = nn.BatchNorm2d(out_channels)
        self.soft = nn.Softmax(-2)
        self.relu = nn.ReLU(inplace=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)
        bn_init(self.bn, 1e-6)

    def forward(self, x):
        y = None
        if self.adaptive:
            A = self.PA
        else:
            A = self.A.cuda(x.get_device())
        for i in range(self.num_subset):
            z = self.convs[i](x, A[i], self.alpha,self.beta)
            y = z + y if y is not None else z
        y = self.bn(y)
        y += self.down(x)
        y = self.relu(y)

        return y


class TCN_GCN_unit(nn.Module):
    def __init__(self, in_channels, out_channels, A, stride=1, residual=True, adaptive=True, kernel_size=None, dilations=[1,2,3,4,5,6,7,8]):
        super(TCN_GCN_unit, self).__init__()
        self.gcn1 = unit_gcn(in_channels, out_channels, A, adaptive=adaptive)

        self.tcn1 = MultiScale_TemporalConv(out_channels, out_channels, kernel_size=kernel_size, stride=stride, dilations=dilations,
                                            residual=False)

        self.relu = nn.ReLU(inplace=True)
        if not residual:
            self.residual = lambda x: 0

        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x

        else:
            self.residual = unit_tcn(in_channels, out_channels, kernel_size=1, stride=stride)

    def forward(self, x):
        y = self.relu(self.tcn1(self.gcn1(x)) + self.residual(x))
        return y


class Model(nn.Module):
    def __init__(self, num_class=11, num_point=15,  graph=None, in_channels=3, adaptive=True, num_channel = 64, temporal_kernel = None, dilations =None):
        super(Model, self).__init__()


        A = graph


        self.num_class = num_class
        self.num_point = num_point
        self.data_bn = nn.BatchNorm1d(in_channels * num_point)

        base_channel = 64

        self.input_map = nn.Sequential(
            nn.Conv2d(in_channels, num_channel//2, 1),
            nn.BatchNorm2d(num_channel//2),
            nn.LeakyReLU(0.1),
        )
        self.diff_map1 = nn.Sequential(
            nn.Conv2d(in_channels, num_channel//8, 1),
            nn.BatchNorm2d(num_channel//8),
            nn.LeakyReLU(0.1),
        )
        self.diff_map2 = nn.Sequential(
            nn.Conv2d(in_channels, num_channel//8, 1),
            nn.BatchNorm2d(num_channel//8),
            nn.LeakyReLU(0.1),
        )
        self.diff_map3 = nn.Sequential(
            nn.Conv2d(in_channels, num_channel//8, 1),
            nn.BatchNorm2d(num_channel//8),
            nn.LeakyReLU(0.1),
        )
        self.diff_map4 = nn.Sequential(
            nn.Conv2d(in_channels, num_channel//8, 1),
            nn.BatchNorm2d(num_channel//8),
            nn.LeakyReLU(0.1),
        )
        assert num_channel <= 64


        self.l1 = TCN_GCN_unit(num_channel, base_channel, A, kernel_size=temporal_kernel, residual=False, adaptive=adaptive,dilations =dilations)
        self.l2 = TCN_GCN_unit(base_channel, base_channel, A,kernel_size=temporal_kernel, adaptive=adaptive,dilations =dilations)
        self.l3 = TCN_GCN_unit(base_channel, base_channel, A,kernel_size=temporal_kernel, adaptive=adaptive,dilations =dilations)
        self.l4 = TCN_GCN_unit(base_channel, base_channel, A,kernel_size=temporal_kernel, adaptive=adaptive,dilations =dilations)
        self.l5 = TCN_GCN_unit(base_channel, base_channel*2, A,kernel_size=temporal_kernel, adaptive=adaptive,dilations =dilations)
        self.l6 = TCN_GCN_unit(base_channel*2, base_channel*2, A,kernel_size=temporal_kernel, adaptive=adaptive,dilations =dilations)
        self.l7 = TCN_GCN_unit(base_channel*2, base_channel*2, A,kernel_size=temporal_kernel, adaptive=adaptive,dilations =dilations)
        self.l8 = TCN_GCN_unit(base_channel*2, base_channel*4, A,kernel_size=temporal_kernel, adaptive=adaptive,dilations =dilations)
        self.l9 = TCN_GCN_unit(base_channel*4, base_channel*4, A,kernel_size=temporal_kernel, adaptive=adaptive,dilations =dilations)
        self.l10 = TCN_GCN_unit(base_channel*4, base_channel*4, A,kernel_size=temporal_kernel, adaptive=adaptive,dilations =dilations)

        self.fc = nn.Conv2d(base_channel*4, self.num_class, kernel_size=(1, self.num_point))
        nn.init.normal_(self.fc.weight, 0, math.sqrt(2. / num_class))
        bn_init(self.data_bn, 1)
        print("using NTVV")

    def forward(self, x):
        N, C, T, V = x.size()

        # x = x.permute(0, 4, 3, 1, 2).contiguous().view(N, M * V * C, T)
        x = x.permute(0, 3, 1, 2).contiguous().view(N, V * C, T)
        x = self.data_bn(x)
        # x = x.view(N, M, V, C, T).permute(0, 1, 3, 4, 2).contiguous().view(N * M, C, T, V)
        x = x.view(N, V, C, T).permute(0, 2, 3, 1).contiguous() # (N, C, T, V)


        # motion infomation

        dif1 = x[:, :, 1:] - x[:, :, 0:-1]
        dif1 = torch.cat([dif1.new(N, C, 1, V).zero_(), dif1], dim=-2)
        dif2 = x[:, :, 2:] - x[:, :, 0:-2]
        dif2 = torch.cat([dif2.new(N, C, 2, V).zero_(), dif2], dim=-2)
        dif3 = x[:, :, :-1] - x[:, :, 1:]
        dif3 = torch.cat([dif3, dif3.new(N, C, 1, V).zero_()], dim=-2)
        dif4 = x[:, :, :-2] - x[:, :, 2:]
        dif4 = torch.cat([dif4, dif4.new(N, C, 2, V).zero_()], dim=-2)

        x = torch.cat((self.input_map(x), self.diff_map1(dif1), self.diff_map2(dif2), self.diff_map3(dif3), self.diff_map4(dif4)), dim = 1)


        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        x = self.l4(x)
        x = self.l5(x)
        x = self.l6(x)
        x = self.l7(x)
        x = self.l8(x)
        x = self.l9(x)
        x = self.l10(x)


        return self.fc(x)

if __name__ == '__main__':
    import openpack_torch as optorch
    from openpack_torch.models.keypoint.graph_new1 import Graph
    g = Graph(strategy='uniform')
    # Ks=3
    # A = optorch.models.keypoint.get_adjacency_matrix(
    #     layout="MSCOCO", hop_size=Ks - 1)

    A = g.A
    print(A.shape)
    # x=torch.zeros((4,2,900,15)).cuda()
    M=Model(graph=A, in_channels=2, num_point=15, temporal_kernel = 7,dilations =[1])
    # out=M(x)
    # print(out.shape)


    # M=Model(graph=A, in_channels=2, num_point=15, temporal_kernel = 7,dilations =[1,2,3,4,5,6,7,8]) #.cuda()
    #
    # from torchstat import stat
    #
    # stat(M, (2, 900, 15))



    # total = sum(p.numel() for p in M.parameters())
    # print(total)
    # print("Total params: %.2fM" % (total / 1e6))

    from fvcore.nn import FlopCountAnalysis, parameter_count_table
    # 创建输入网络的tensor
    tensor = (torch.rand(1, 2, 900, 15),)
    # 分析FLOPs
    flops = FlopCountAnalysis(M, tensor)
    print("FLOPs: ", flops.total())




