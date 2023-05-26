import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
import torchsparse
from torchsparse import PointTensor, SparseTensor
import torchsparse.nn as spnn

import os
import sys
import math
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(os.path.dirname(currentdir)))
sys.path.insert(0, parentdir) 
from core.models.utils import * #initial_voxelize, point_to_voxel, voxel_to_point
from torchsparse import PointTensor
from time import time
from torch.cuda import synchronize

__all__ = ['ELKEncoder']


class BasicConvolutionBlock(nn.Module):

    def __init__(self, inc, outc, ks=3, stride=1, dilation=1):
        super().__init__()
        self.net = nn.Sequential(
            spnn.Conv3d(inc,
                        outc,
                        kernel_size=ks,
                        dilation=dilation,
                        stride=stride),
            spnn.BatchNorm(outc),
            spnn.ReLU(True),
        )

    def forward(self, x):
        out = self.net(x)
        return out


class BasicDeconvolutionBlock(nn.Module):

    def __init__(self, inc, outc, ks=3, stride=1):
        super().__init__()
        self.net = nn.Sequential(
            spnn.Conv3d(inc,
                        outc,
                        kernel_size=ks,
                        stride=stride,
                        transposed=True),
            spnn.BatchNorm(outc),
            spnn.ReLU(True),
        )

    def forward(self, x):
        out = self.net(x)
        return out


class ResidualBlock(nn.Module):

    def __init__(self, inc, outc, ks=3, stride=1, dilation=1):
        super().__init__()
        self.net = nn.Sequential(
            spnn.Conv3d(inc,
                        outc,
                        kernel_size=ks,
                        dilation=dilation,
                        stride=stride),
            spnn.BatchNorm(outc),
            spnn.ReLU(True),
            spnn.Conv3d(outc, outc, kernel_size=ks, dilation=dilation,
                        stride=1),
            spnn.BatchNorm(outc),
        )

        if inc == outc and stride == 1:
            self.downsample = nn.Sequential()
        else:
            self.downsample = nn.Sequential(
                spnn.Conv3d(inc, outc, kernel_size=1, dilation=1,
                            stride=stride),
                spnn.BatchNorm(outc),
            )

        self.relu = spnn.ReLU(True)

    def forward(self, x):
        out = self.relu(self.net(x) + self.downsample(x))
        return out


class ELKBlock(nn.Module):
    def __init__(self, inc, outc, groups=1, baseop='cos_x'):
        super().__init__()
        self.inc = inc
        self.outc = outc
        self.groups = groups
        assert inc % self.groups == 0
        self.baseop = baseop
        
        assert self.baseop in ['cos', 'sin', 'cos_x']

        if self.baseop == 'cos_x':
            self.alpha = nn.Parameter(torch.ones(1, self.inc//self.groups).float(), requires_grad = True)
        
        self.pos_weight = nn.Sequential(
            nn.Linear(3, self.inc//self.groups, bias=False),
        )

        self.pre_mix = nn.Sequential(
            nn.Linear(self.inc, self.inc, bias = False),
            nn.LayerNorm(self.inc, eps=1e-6)
        )
        self.local_mix = nn.Sequential(
            spnn.Conv3d(self.inc, self.inc, kernel_size=3, dilation=1,
                            stride=1),
        )
        self.norm_local = nn.LayerNorm(self.inc, eps=1e-6)
        self.norm = nn.LayerNorm(self.inc, eps=1e-6)
        self.activate = nn.ReLU(True)
    
    def forward(self, st, s, r):
        '''
            st: SparseTensor
            s: scale of block
            r: scale of block query

        '''
        F_ori = st.F
        F_input = self.pre_mix(st.F)
        local_mix = self.local_mix(st)
            
        if self.baseop == 'sin':
            pos_weight = self.pos_weight(st.C[:,:3].float())
            pos_weight = pos_weight.repeat([1,self.groups])

            pos_weight_sin = torch.sin(pos_weight)
            pos_weight_cos = torch.cos(pos_weight)
            F_weighted_sin = F_input*pos_weight_sin
            F_weighted_cos = F_input*pos_weight_cos
            st.F = torch.cat([F_weighted_sin, F_weighted_cos], dim=1).contiguous()

            aux_st, idx, counts = voxel_to_aux(st, s)
            voxel_st = aux_to_voxel(aux_st, st, idx, counts, r)
            
            new_st_F = voxel_st.F[:,:self.inc]*pos_weight_cos - voxel_st.F[:,self.inc:]*pos_weight_sin

        elif self.baseop == 'cos':
            pos_weight = self.pos_weight(st.C[:,:3].float())
            pos_weight = pos_weight.repeat([1,self.groups])

            pos_weight_sin = torch.sin(pos_weight)
            pos_weight_cos = torch.cos(pos_weight)
            F_weighted_sin = F_input*pos_weight_sin
            F_weighted_cos = F_input*pos_weight_cos
            st.F = torch.cat([F_weighted_cos, F_weighted_sin], dim=1).contiguous()

            aux_st, idx, counts = voxel_to_aux(st, s)
            voxel_st = aux_to_voxel(aux_st, st, idx, counts, r)
            new_st_F = voxel_st.F[:,:self.inc]*pos_weight_cos + voxel_st.F[:,self.inc:]*pos_weight_sin

        elif self.baseop == 'cos_x':
            pos_weight = self.pos_weight(st.C[:,:3].float()/st.s[0])* self.alpha

            pos_weight_sin = torch.sin(pos_weight)
            pos_weight_cos = torch.cos(pos_weight)
            F_weighted_sin = F_input*pos_weight_sin
            F_weighted_cos = F_input*pos_weight_cos
            F_weighted_linear = F_input*pos_weight
            st.F = torch.cat([F_weighted_cos, F_weighted_sin, F_weighted_linear], dim=1).contiguous()

            aux_st, idx, counts = voxel_to_aux(st, s)
            voxel_st = aux_to_voxel(aux_st, st, idx, counts, r)
            new_st_F = voxel_st.F[:,:self.inc]*pos_weight_cos + voxel_st.F[:,self.inc:2*self.inc]*pos_weight_sin + (voxel_st.F[:,2*self.inc:]-F_weighted_linear)

        new_st_F = self.norm(new_st_F)
        local_F = self.norm_local(local_mix.F)
        
        new_st_F = self.activate(new_st_F+local_F)

        voxel_st.F = new_st_F

        return voxel_st


class ELKEncoder(nn.Module):

    def __init__(self, **kwargs):
        super().__init__()

        self.kwargs = kwargs
        cr = kwargs.get('cr', 1.0)
        baseop = kwargs.get('baseop')
        cs = [64, 64, 64, 64, 64, 64, 64, 64, 64]
        cs = [int(cr * x) for x in cs]
        self.run_up = kwargs.get('run_up', True)

        self.stem = nn.Sequential(
            spnn.Conv3d(4, cs[0], kernel_size=3, stride=1),
            spnn.BatchNorm(cs[0]), spnn.ReLU(True),
            spnn.Conv3d(cs[0], cs[0], kernel_size=3, stride=1),
            spnn.BatchNorm(cs[0]), spnn.ReLU(True)
        )

        self.down1 = nn.Sequential(
            BasicConvolutionBlock(cs[0], cs[0], ks=2, stride=2, dilation=1),
        )

        self.stage1 = nn.Sequential(
            ResidualBlock(cs[0], cs[1], ks=3, stride=1, dilation=1),
            ResidualBlock(cs[1], cs[1], ks=3, stride=1, dilation=1),
        )

        self.stage1_tail = nn.Sequential(
            spnn.Conv3d(cs[1], cs[1], kernel_size=3, stride=1),
            spnn.BatchNorm(cs[1]),
        )
        self.elk1 = ELKBlock(cs[0], cs[0], self.kwargs.get('groups'), baseop=baseop)
        self.elk1_tail = nn.Sequential(
            spnn.Conv3d(cs[0], cs[1], kernel_size=3, stride=1),
            spnn.BatchNorm(cs[1]),
        )

        self.activate1 = nn.ReLU(True)

        self.down2 = nn.Sequential(
            BasicConvolutionBlock(cs[1], cs[1], ks=2, stride=2, dilation=1),
        )

        self.stage2 = nn.Sequential(
            ResidualBlock(cs[1], cs[2], ks=3, stride=1, dilation=1),
            ResidualBlock(cs[2], cs[2], ks=3, stride=1, dilation=1)
        )

        self.stage2_tail = nn.Sequential(
            spnn.Conv3d(cs[2], cs[2], kernel_size=3, stride=1),
            spnn.BatchNorm(cs[2]),
        )
        self.elk2 = ELKBlock(cs[1], cs[1], self.kwargs.get('groups'), baseop=baseop)
        self.elk2_tail = nn.Sequential(
            spnn.Conv3d(cs[1], cs[2], kernel_size=3, stride=1),
            spnn.BatchNorm(cs[2]),
        )
        self.activate2 = nn.ReLU(True)

        self.down3 = nn.Sequential(
            BasicConvolutionBlock(cs[2], cs[2], ks=2, stride=2, dilation=1),
        )

        self.stage3 = nn.Sequential(
            ResidualBlock(cs[2], cs[3], ks=3, stride=1, dilation=1),
            ResidualBlock(cs[3], cs[3], ks=3, stride=1, dilation=1),
        )

        self.stage3_tail = nn.Sequential(
            spnn.Conv3d(cs[3], cs[3], kernel_size=3, stride=1),
            spnn.BatchNorm(cs[3]),
        )
        self.elk3 = ELKBlock(cs[2], cs[2], self.kwargs.get('groups'), baseop=baseop)
        self.elk3_tail = nn.Sequential(
            spnn.Conv3d(cs[2], cs[3], kernel_size=3, stride=1),
            spnn.BatchNorm(cs[3]),
        )
        self.activate3 = nn.ReLU(True)

        self.down4 = nn.Sequential(
            BasicConvolutionBlock(cs[3], cs[3], ks=2, stride=2, dilation=1),
        )
        
        self.stage4 = nn.Sequential(
            ResidualBlock(cs[3], cs[4], ks=3, stride=1, dilation=1),
            ResidualBlock(cs[4], cs[4], ks=3, stride=1, dilation=1),
        )

        self.stage4_tail = nn.Sequential(
            spnn.Conv3d(cs[4], cs[4], kernel_size=3, stride=1),
            spnn.BatchNorm(cs[4]),
        )
        self.elk4 = ELKBlock(cs[3], cs[3], self.kwargs.get('groups'), baseop=baseop)
        self.elk4_tail = nn.Sequential(
            spnn.Conv3d(cs[3], cs[4], kernel_size=3, stride=1),
            spnn.BatchNorm(cs[4]),
        )
        self.activate4 = nn.ReLU(True)


        self.up1 = nn.ModuleList([
            BasicDeconvolutionBlock(cs[4], cs[5], ks=2, stride=2),
            nn.Sequential(
                ResidualBlock(cs[5] + cs[3], cs[5], ks=3, stride=1, dilation=1),
                ResidualBlock(cs[5], cs[5], ks=3, stride=1, dilation=1),
            )
        ])

        self.up2 = nn.ModuleList([
            BasicDeconvolutionBlock(cs[5], cs[6], ks=2, stride=2),
            nn.Sequential(
                ResidualBlock(cs[6] + cs[2], cs[6], ks=3, stride=1, dilation=1),
                ResidualBlock(cs[6], cs[6], ks=3, stride=1, dilation=1),
            )
        ])

        self.up3 = nn.ModuleList([
            BasicDeconvolutionBlock(cs[6], cs[7], ks=2, stride=2),
            nn.Sequential(
                ResidualBlock(cs[7] + cs[1], cs[7], ks=3, stride=1, dilation=1),
                ResidualBlock(cs[7], cs[7], ks=3, stride=1, dilation=1),
            )
        ])


        self.up4 = nn.ModuleList([
            BasicDeconvolutionBlock(cs[7], cs[8], ks=2, stride=2),
            nn.Sequential(
                ResidualBlock(cs[8] + cs[0], cs[8], ks=3, stride=1, dilation=1),
                ResidualBlock(cs[8], cs[8], ks=3, stride=1, dilation=1),
            )
        ])


        self.classifier = nn.Sequential(
                nn.Conv1d(in_channels=cs[8]*5, out_channels=120, kernel_size=1, groups=5),
                nn.ReLU(True),
                nn.Conv1d(in_channels=120, out_channels=kwargs['num_classes'], kernel_size=1, groups=1),
            )
        
        self.weight_initialization()

    def weight_initialization(self):
        for m in self.modules():
            if isinstance(m, (nn.BatchNorm1d, nn.LayerNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            # elif isinstance(m, nn.Linear):
            #     nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))

    def forward(self, x):
        
        # stem
        x0 = self.stem(x)

        s = self.kwargs.get('s')
        r = self.kwargs.get('r')
        # layer 1
        x1_0 = self.down1(x0)
        x1 = self.stage1_tail(self.stage1(x1_0))
        x1_lk = self.elk1_tail(self.elk1(x1_0, x1_0.s[0]*s, r))
        x1.F = self.activate1(x1.F+x1_lk.F)        

        # layer 2
        x2_0 = self.down2(x1)
        x2 = self.stage2_tail(self.stage2(x2_0))
        x2_lk = self.elk2_tail(self.elk2(x2_0, x2_0.s[0]*s, r))
        x2.F = self.activate2(x2.F+x2_lk.F)

        # layer 3
        x3_0 = self.down3(x2)
        x3 = self.stage3_tail(self.stage3(x3_0))
        x3_lk = self.elk3_tail(self.elk3(x3_0, x3_0.s[0]*s, r))
        x3.F = self.activate3(x3.F+x3_lk.F)

        # layer 4
        x4_0 = self.down4(x3)
        x4 = self.stage4_tail(self.stage4(x4_0))
        x4_lk = self.elk4_tail(self.elk4(x4_0, x4_0.s[0]*s, r))
        x4.F = self.activate4(x4.F+x4_lk.F)
        
        y4 = upsample_voxel(x4, x0)
        y3 = upsample_voxel(x3, x0)
        y2 = upsample_voxel(x2, x0)
        y1 = upsample_voxel(x1, x0)

        
        F_cat = torch.cat([y4.F, y3.F, y2.F, y1.F, x0.F], dim=1)
        F_cat = F_cat.unsqueeze(dim=0).permute(0,2,1)
        out = self.classifier(F_cat)
        out = out.squeeze(dim=0).T

        return out

if __name__ == '__main__':
    N = 80000
    num_per_pc = 80000
    assert N % num_per_pc == 0
    max_depth = 4
    ratio = 1.0
    
    print('#points: {}'.format(N))
    
    # data preparation
    feats = torch.rand(N, 4).cuda()
    coords = torch.rand(N, 4).cuda()
    coords[:,:3] *= 100
    for i in range(N // num_per_pc):
        coords[num_per_pc * i : num_per_pc * (i + 1), 3] = i
    
    pt = PointTensor(feats=feats, coords=coords)
    st = initial_voxelize(pt, 1, 1)

    model = ELKEncoder(num_classes=19).cuda().eval()
    for i in range(50):
        out = model(st)
        
    