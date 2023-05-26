import numpy as np
# try:
import spconv.pytorch as spconv 
from spconv.pytorch import ops
from spconv.pytorch import SparseConv3d, SubMConv3d
# except: 
#     import spconv 
#     from spconv import ops
#     from spconv import SparseConv3d, SubMConv3d

from torch import nn
from torch.nn import functional as F

from ..registry import BACKBONES
from ..utils import build_norm_layer
from ..utils.ts_elk import spconv2ts, ts2spconv, TSELKBlock, TSELKBlock_no_tail_norm

import os
def uniquify(path):
    filename, extension = os.path.splitext(path)
    counter = 1

    while os.path.exists(path):
        path = filename + str(counter) + extension
        counter += 1

    return path


def replace_feature(out:spconv.SparseConvTensor, new_features):
    if "replace_feature" in out.__dir__():
        # spconv 2.x behaviour
        return out.replace_feature(new_features)
    else:
        out.features = new_features
        return out

def conv3x3(in_planes, out_planes, stride=1, indice_key=None, bias=True):
    """3x3 convolution with padding"""
    return spconv.SubMConv3d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=bias,
        indice_key=indice_key,
    )


def conv1x1(in_planes, out_planes, stride=1, indice_key=None, bias=True):
    """1x1 convolution"""
    return spconv.SubMConv3d(
        in_planes,
        out_planes,
        kernel_size=1,
        stride=stride,
        padding=1,
        bias=bias,
        indice_key=indice_key,
    )


class SparseBasicBlock(spconv.SparseModule):
    expansion = 1

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        norm_cfg=None,
        downsample=None,
        indice_key=None,
    ):
        super(SparseBasicBlock, self).__init__()

        if norm_cfg is None:
            norm_cfg = dict(type="BN1d", eps=1e-3, momentum=0.01)

        bias = norm_cfg is not None

        self.conv1 = conv3x3(inplanes, planes, stride, indice_key=indice_key, bias=bias)
        self.bn1 = build_norm_layer(norm_cfg, planes)[1]
        self.relu = nn.ReLU()
        self.conv2 = conv3x3(planes, planes, indice_key=indice_key, bias=bias)
        self.bn2 = build_norm_layer(norm_cfg, planes)[1]
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = replace_feature(out, self.bn1(out.features))
        out = replace_feature(out, self.relu(out.features))

        out = self.conv2(out)
        out = replace_feature(out, self.bn2(out.features))

        if self.downsample is not None:
            identity = self.downsample(x)

        out = replace_feature(out, out.features + identity.features)
        out = replace_feature(out, self.relu(out.features))

        return out


class TSELKBlockPara(spconv.SparseModule):
    expansion = 1

    def __init__(
        self,
        inplanes,
        planes,
        large_ks=2,
        norm_cfg=None,
        indice_key=None,
    ):
        assert inplanes == planes, "for ELKBlock, in and out channels must be the same"

        super(TSELKBlockPara, self).__init__()

        if norm_cfg is None:
            norm_cfg = dict(type="BN1d", eps=1e-3, momentum=0.01)

        bias = norm_cfg is not None

        self.conv1 = conv3x3(inplanes, planes, indice_key=indice_key, bias=bias)
        self.bn1 = build_norm_layer(norm_cfg, planes)[1]
        self.relu = nn.ReLU()
        self.conv2 = conv3x3(planes, planes, indice_key=indice_key, bias=bias)
        self.bn2 = build_norm_layer(norm_cfg, planes)[1]
        self.elk = TSELKBlock_no_tail_norm(inplanes, planes)

    def forward(self, x):
        identity = x

        out1 = self.conv1(x)
        out1 = replace_feature(out1, self.bn1(out1.features))
        out1 = replace_feature(out1, self.relu(out1.features))
        out1 = self.conv2(out1)
        out1 = replace_feature(out1, self.bn2(out1.features))

        out2 = self.elk(x, stride=7)
        out2 = replace_feature(out2, self.bn2(out2.features))

        out = replace_feature(out1, out1.features + out2.features)
        out = replace_feature(out, self.relu(out.features))

        return out


@BACKBONES.register_module
class SpMiddleResNetFHD(nn.Module):
    def __init__(
        self, num_input_features=128, norm_cfg=None, name="SpMiddleResNetFHD", **kwargs
    ):
        super(SpMiddleResNetFHD, self).__init__()
        self.name = name

        self.dcn = None
        self.zero_init_residual = False

        if norm_cfg is None:
            norm_cfg = dict(type="BN1d", eps=1e-3, momentum=0.01)

        # input: # [1600, 1200, 41]
        self.conv_input = spconv.SparseSequential(
            SubMConv3d(num_input_features, 16, 3, bias=False, indice_key="res0"),
            build_norm_layer(norm_cfg, 16)[1],
            nn.ReLU(inplace=True)
        )

        self.conv1 = spconv.SparseSequential(        
            SparseBasicBlock(16, 16, norm_cfg=norm_cfg, indice_key="res0"),
            SparseBasicBlock(16, 16, norm_cfg=norm_cfg, indice_key="res0"),
        )
        self.conv2 = spconv.SparseSequential(
            SparseConv3d(
                16, 32, 3, 2, padding=1, bias=False
            ),  # [1600, 1200, 41] -> [800, 600, 21]
            build_norm_layer(norm_cfg, 32)[1],
            nn.ReLU(inplace=True),
            SparseBasicBlock(32, 32, norm_cfg=norm_cfg, indice_key="res1"),
            SparseBasicBlock(32, 32, norm_cfg=norm_cfg, indice_key="res1"),
        )
        self.conv3 = spconv.SparseSequential(
            SparseConv3d(
                32, 64, 3, 2, padding=1, bias=False
            ),  # [800, 600, 21] -> [400, 300, 11]
            build_norm_layer(norm_cfg, 64)[1],
            nn.ReLU(inplace=True),
            SparseBasicBlock(64, 64, norm_cfg=norm_cfg, indice_key="res2"),
            SparseBasicBlock(64, 64, norm_cfg=norm_cfg, indice_key="res2"),
        )
        self.conv4 = spconv.SparseSequential(
            SparseConv3d(
                64, 128, 3, 2, padding=[0, 1, 1], bias=False
            ),  # [400, 300, 11] -> [200, 150, 5]
            build_norm_layer(norm_cfg, 128)[1],
            nn.ReLU(inplace=True),
            SparseBasicBlock(128, 128, norm_cfg=norm_cfg, indice_key="res3"),
            SparseBasicBlock(128, 128, norm_cfg=norm_cfg, indice_key="res3"),
        )

        self.extra_conv = spconv.SparseSequential(
            SparseConv3d(
                128, 128, (3, 1, 1), (2, 1, 1), bias=False
            ),  # [200, 150, 5] -> [200, 150, 2]
            build_norm_layer(norm_cfg, 128)[1],
            nn.ReLU(),
        )

    def forward(self, voxel_features, coors, batch_size, input_shape):

        # input: # [41, 1440, 1440]
        sparse_shape = np.array(input_shape[::-1]) + [1, 0, 0]

        coors = coors.int()
        ret = spconv.SparseConvTensor(voxel_features, coors, sparse_shape, batch_size)

        x = self.conv_input(ret) # x: [41, 1440, 1440]

        x_conv1 = self.conv1(x) # x_conv1: [41, 1440, 1440]
        x_conv2 = self.conv2(x_conv1) # x_conv2: [21, 720, 720]
        x_conv3 = self.conv3(x_conv2) # x_conv3: [11, 360, 360]
        x_conv4 = self.conv4(x_conv3) # x_conv4: [5, 180, 180]
        ret = self.extra_conv(x_conv4) # ret: [2, 180, 180]

        ret = ret.dense()

        N, C, D, H, W = ret.shape # 4, 128, 2, 180, 180
        ret = ret.view(N, C * D, H, W)

        multi_scale_voxel_features = {
            'conv1': x_conv1,
            'conv2': x_conv2,
            'conv3': x_conv3,
            'conv4': x_conv4,
        }

        return ret, multi_scale_voxel_features


# ELK v1 inserts serial ELK blocks among conv layers serially
@BACKBONES.register_module
class SpMiddleResNetFHDELKv1(nn.Module):
    def __init__(
        self, num_input_features=128, norm_cfg=None, name="SpMiddleResNetFHDELKv1", **kwargs
    ):
        super(SpMiddleResNetFHDELKv1, self).__init__()
        self.name = name

        self.dcn = None
        self.zero_init_residual = False

        if norm_cfg is None:
            norm_cfg = dict(type="BN1d", eps=1e-3, momentum=0.01)

        # input: # [1600, 1200, 41]
        self.conv_input = spconv.SparseSequential(
            SubMConv3d(num_input_features, 16, 3, bias=False, indice_key="res0"),
            build_norm_layer(norm_cfg, 16)[1],
            nn.ReLU(inplace=True)
        )

        self.conv1 = spconv.SparseSequential(        
            SparseBasicBlock(16, 16, norm_cfg=norm_cfg, indice_key="res0"),
            SparseBasicBlock(16, 16, norm_cfg=norm_cfg, indice_key="res0"),
        )
        self.elk1 = TSELKBlock(16, 16)

        self.conv2 = spconv.SparseSequential(
            SparseConv3d(
                16, 32, 3, 2, padding=1, bias=False
            ),  # [1600, 1200, 41] -> [800, 600, 21]
            build_norm_layer(norm_cfg, 32)[1],
            nn.ReLU(inplace=True),
            SparseBasicBlock(32, 32, norm_cfg=norm_cfg, indice_key="res1"),
            SparseBasicBlock(32, 32, norm_cfg=norm_cfg, indice_key="res1"),
        )
        self.elk2 = TSELKBlock(32, 32)

        self.conv3 = spconv.SparseSequential(
            SparseConv3d(
                32, 64, 3, 2, padding=1, bias=False
            ),  # [800, 600, 21] -> [400, 300, 11]
            build_norm_layer(norm_cfg, 64)[1],
            nn.ReLU(inplace=True),
            SparseBasicBlock(64, 64, norm_cfg=norm_cfg, indice_key="res2"),
            SparseBasicBlock(64, 64, norm_cfg=norm_cfg, indice_key="res2"),
        )
        self.elk3 = TSELKBlock(64, 64)

        self.conv4 = spconv.SparseSequential(
            SparseConv3d(
                64, 128, 3, 2, padding=[0, 1, 1], bias=False
            ),  # [400, 300, 11] -> [200, 150, 5]
            build_norm_layer(norm_cfg, 128)[1],
            nn.ReLU(inplace=True),
            SparseBasicBlock(128, 128, norm_cfg=norm_cfg, indice_key="res3"),
            SparseBasicBlock(128, 128, norm_cfg=norm_cfg, indice_key="res3"),
        )
        self.elk4 = TSELKBlock(128, 128)


        self.extra_conv = spconv.SparseSequential(
            SparseConv3d(
                128, 128, (3, 1, 1), (2, 1, 1), bias=False
            ),  # [200, 150, 5] -> [200, 150, 2]
            build_norm_layer(norm_cfg, 128)[1],
            nn.ReLU(),
        )

    def forward(self, voxel_features, coors, batch_size, input_shape):

        # input: # [41, 1600, 1408]
        sparse_shape = np.array(input_shape[::-1]) + [1, 0, 0]

        coors = coors.int()
        ret = spconv.SparseConvTensor(voxel_features, coors, sparse_shape, batch_size)

        x = self.conv_input(ret)

        x_conv1 = self.conv1(x)
        x_conv1 = self.elk1(x_conv1, stride=7)

        x_conv2 = self.conv2(x_conv1)
        x_conv2 = self.elk2(x_conv2, stride=7)

        x_conv3 = self.conv3(x_conv2)
        x_conv3 = self.elk3(x_conv3, stride=7)

        x_conv4 = self.conv4(x_conv3)
        x_conv4 = self.elk4(x_conv4, stride=7)

        ret = self.extra_conv(x_conv4)

        ret = ret.dense()

        N, C, D, H, W = ret.shape
        ret = ret.view(N, C * D, H, W)

        multi_scale_voxel_features = {
            'conv1': x_conv1,
            'conv2': x_conv2,
            'conv3': x_conv3,
            'conv4': x_conv4,
        }

        return ret, multi_scale_voxel_features


# ELK v2 inserts parallel ELK blocks alongside conv block (as in LK3D)
@BACKBONES.register_module
class SpMiddleResNetFHDELKv2(nn.Module):
    def __init__(
        self, num_input_features=128, norm_cfg=None, name="SpMiddleResNetFHDELKv2", **kwargs
    ):
        super(SpMiddleResNetFHDELKv2, self).__init__()
        self.name = name

        self.dcn = None
        self.zero_init_residual = False

        if norm_cfg is None:
            norm_cfg = dict(type="BN1d", eps=1e-3, momentum=0.01)

        # input: # [1600, 1200, 41]
        self.conv_input = spconv.SparseSequential(
            SubMConv3d(num_input_features, 16, 3, bias=False, indice_key="res0"),
            build_norm_layer(norm_cfg, 16)[1],
            nn.ReLU(inplace=True)
        )

        self.conv1 = spconv.SparseSequential(        
            TSELKBlockPara(16, 16, norm_cfg=norm_cfg, indice_key="res0"),
            TSELKBlockPara(16, 16, norm_cfg=norm_cfg, indice_key="res0"),
        )

        self.conv2 = spconv.SparseSequential(
            SparseConv3d(
                16, 32, 3, 2, padding=1, bias=False
            ),  # [1600, 1200, 41] -> [800, 600, 21]
            build_norm_layer(norm_cfg, 32)[1],
            nn.ReLU(inplace=True),
            TSELKBlockPara(32, 32, norm_cfg=norm_cfg, indice_key="res1"),
            TSELKBlockPara(32, 32, norm_cfg=norm_cfg, indice_key="res1"),
        )

        self.conv3 = spconv.SparseSequential(
            SparseConv3d(
                32, 64, 3, 2, padding=1, bias=False
            ),  # [800, 600, 21] -> [400, 300, 11]
            build_norm_layer(norm_cfg, 64)[1],
            nn.ReLU(inplace=True),
            TSELKBlockPara(64, 64, norm_cfg=norm_cfg, indice_key="res2"),
            TSELKBlockPara(64, 64, norm_cfg=norm_cfg, indice_key="res2"),
        )

        self.conv4 = spconv.SparseSequential(
            SparseConv3d(
                64, 128, 3, 2, padding=[0, 1, 1], bias=False
            ),  # [400, 300, 11] -> [200, 150, 5]
            build_norm_layer(norm_cfg, 128)[1],
            nn.ReLU(inplace=True),
            SparseBasicBlock(128, 128, norm_cfg=norm_cfg, indice_key="res3"),
            SparseBasicBlock(128, 128, norm_cfg=norm_cfg, indice_key="res3"),
        )

        self.extra_conv = spconv.SparseSequential(
            SparseConv3d(
                128, 128, (3, 1, 1), (2, 1, 1), bias=False
            ),  # [200, 150, 5] -> [200, 150, 2]
            build_norm_layer(norm_cfg, 128)[1],
            nn.ReLU(),
        )

    def forward(self, voxel_features, coors, batch_size, input_shape):

        # input: # [41, 1600, 1408]
        sparse_shape = np.array(input_shape[::-1]) + [1, 0, 0]

        coors = coors.int()
        ret = spconv.SparseConvTensor(voxel_features, coors, sparse_shape, batch_size)

        x = self.conv_input(ret)

        x_conv1 = self.conv1(x)
        x_conv2 = self.conv2(x_conv1)
        x_conv3 = self.conv3(x_conv2)
        x_conv4 = self.conv4(x_conv3)
        ret = self.extra_conv(x_conv4)

        ret = ret.dense()

        N, C, D, H, W = ret.shape
        ret = ret.view(N, C * D, H, W)

        multi_scale_voxel_features = {
            'conv1': x_conv1,
            'conv2': x_conv2,
            'conv3': x_conv3,
            'conv4': x_conv4,
        }

        return ret, multi_scale_voxel_features


# ELK v3 inserts parallel ELK blocks alongside the whole conv stage
@BACKBONES.register_module
class SpMiddleResNetFHDELKv3(nn.Module):
    def __init__(
        self, num_input_features=128, norm_cfg=None, name="SpMiddleResNetFHDELKv3", **kwargs
    ):
        super(SpMiddleResNetFHDELKv3, self).__init__()
        self.name = name

        self.dcn = None
        self.zero_init_residual = False

        if norm_cfg is None:
            norm_cfg = dict(type="BN1d", eps=1e-3, momentum=0.01)

        self.planes = [16, 32, 64, 128]
        # self.planes = [16, 32, 32, 128]

        self.block_sz = 7

        # input: # [1600, 1200, 41]
        self.conv_input = spconv.SparseSequential(
            SubMConv3d(num_input_features, self.planes[0], 3, bias=False, indice_key="res0"),
            build_norm_layer(norm_cfg, self.planes[0])[1],
            nn.ReLU(inplace=True)
        )

        self.conv1 = spconv.SparseSequential(        
            SparseBasicBlock(self.planes[0], self.planes[0], norm_cfg=norm_cfg, indice_key="res1"),
            SparseBasicBlock(self.planes[0], self.planes[0], norm_cfg=norm_cfg, indice_key="res1"),
        )
        self.conv1_tail = spconv.SparseSequential(
            SubMConv3d(self.planes[0], self.planes[0], 3, bias=False, indice_key="res1_tail"),
            build_norm_layer(norm_cfg, self.planes[0])[1],
        )
        self.elk1 = TSELKBlock(self.planes[0], self.planes[0])
        self.elk1_tail = spconv.SparseSequential(
            SubMConv3d(self.planes[0], self.planes[0], 3, bias=False, indice_key="elk1_tail"),
            build_norm_layer(norm_cfg, self.planes[0])[1],
        )
        self.act1 = nn.ReLU(inplace=True)

        self.down2 = spconv.SparseSequential(
            SparseConv3d(
                self.planes[0], self.planes[1], 3, 2, padding=1, bias=False
            ),  # [1600, 1200, 41] -> [800, 600, 21]
            build_norm_layer(norm_cfg, self.planes[1])[1],
            nn.ReLU(inplace=True),
        )
        self.conv2 = spconv.SparseSequential(
            SparseBasicBlock(self.planes[1], self.planes[1], norm_cfg=norm_cfg, indice_key="res2"),
            SparseBasicBlock(self.planes[1], self.planes[1], norm_cfg=norm_cfg, indice_key="res2"),
        )
        self.conv2_tail = spconv.SparseSequential(
            SubMConv3d(self.planes[1], self.planes[1], 3, bias=False, indice_key="res2_tail"),
            build_norm_layer(norm_cfg, 32)[1],
        )
        self.elk2 = TSELKBlock(self.planes[1], self.planes[1])
        self.elk2_tail = spconv.SparseSequential(
            SubMConv3d(self.planes[1], self.planes[1], 3, bias=False, indice_key="elk2_tail"),
            build_norm_layer(norm_cfg, self.planes[1])[1],
        )
        self.act2 = nn.ReLU(inplace=True)

        self.down3 = spconv.SparseSequential(
            SparseConv3d(
                self.planes[1], self.planes[2], 3, 2, padding=1, bias=False
            ),  # [800, 600, 21] -> [400, 300, 11]
            build_norm_layer(norm_cfg, self.planes[2])[1],
            nn.ReLU(inplace=True),
        )
        self.conv3 = spconv.SparseSequential(
            SparseBasicBlock(self.planes[2], self.planes[2], norm_cfg=norm_cfg, indice_key="res3"),
            SparseBasicBlock(self.planes[2], self.planes[2], norm_cfg=norm_cfg, indice_key="res3"),
        )
        self.conv3_tail = spconv.SparseSequential(
            SubMConv3d(self.planes[2], self.planes[2], 3, bias=False, indice_key="res3_tail"),
            build_norm_layer(norm_cfg, self.planes[2])[1],
        )
        self.elk3 = TSELKBlock(self.planes[2], self.planes[2])
        self.elk3_tail = spconv.SparseSequential(
            SubMConv3d(self.planes[2], self.planes[2], 3, bias=False, indice_key="elk3_tail"),
            build_norm_layer(norm_cfg, self.planes[2])[1],
        )
        self.act3 = nn.ReLU(inplace=True)

        self.down4 = spconv.SparseSequential(
            SparseConv3d(
                self.planes[2], self.planes[3], 3, 2, padding=[0, 1, 1], bias=False
            ),  # [400, 300, 11] -> [200, 150, 5]
            build_norm_layer(norm_cfg, 128)[1],
            nn.ReLU(inplace=True),
        )
        self.conv4 = spconv.SparseSequential(
            SparseBasicBlock(self.planes[3], self.planes[3], norm_cfg=norm_cfg, indice_key="res4"),
            SparseBasicBlock(self.planes[3], self.planes[3], norm_cfg=norm_cfg, indice_key="res4"),
        )
        self.conv4_tail = spconv.SparseSequential(
            SubMConv3d(self.planes[3], self.planes[3], 3, bias=False, indice_key="res4_tail"),
            build_norm_layer(norm_cfg, self.planes[3])[1],
        )
        self.elk4 = TSELKBlock(self.planes[3], self.planes[3])
        self.elk4_tail = spconv.SparseSequential(
            SubMConv3d(self.planes[3], self.planes[3], 3, bias=False, indice_key="elk4_tail"),
            build_norm_layer(norm_cfg, self.planes[3])[1],
        )
        self.act4 = nn.ReLU(inplace=True)

        self.extra_conv = spconv.SparseSequential(
            SparseConv3d(
                self.planes[3], self.planes[3], (3, 1, 1), (2, 1, 1), bias=False
            ),  # [200, 150, 5] -> [200, 150, 2]
            build_norm_layer(norm_cfg, self.planes[3])[1],
            nn.ReLU(),
        )

    def forward(self, voxel_features, coors, batch_size, input_shape):

        # input: # [41, 1600, 1408]
        sparse_shape = np.array(input_shape[::-1]) + [1, 0, 0]

        # export PYTHONPATH="${PWD}:${PWD}/nuscenes-devkit/python-sdk"
        # import pickle
        # with open(uniquify('save_pkl/voxel_features.pkl'), 'wb') as f:
        #     pickle.dump(voxel_features, f)
        # with open(uniquify('save_pkl/coors.pkl'), 'wb') as f:
        #     pickle.dump(coors, f)

        coors = coors.int()
        ret = spconv.SparseConvTensor(voxel_features, coors, sparse_shape, batch_size)

        x = self.conv_input(ret)

        x_conv1 = self.conv1_tail(self.conv1(x))
        x_lk1 = self.elk1_tail(self.elk1(x, self.block_sz))
        # x_conv1 = self.conv1(x)
        # x_lk1 = self.elk1(x, 7)
        x_conv1 = replace_feature(x_conv1, self.act1(x_conv1.features + x_lk1.features))

        x_down2 = self.down2(x_conv1)
        x_conv2 = self.conv2_tail(self.conv2(x_down2))
        x_lk2 = self.elk2_tail(self.elk2(x_down2, self.block_sz))
        # x_conv2 = self.conv2(x_down2)
        # x_lk2 = self.elk2(x_down2, 7)
        x_conv2 = replace_feature(x_conv2, self.act2(x_conv2.features + x_lk2.features))
        
        x_down3 = self.down3(x_conv2)
        x_conv3 = self.conv3_tail(self.conv3(x_down3))
        x_lk3 = self.elk3_tail(self.elk3(x_down3, self.block_sz))
        # x_conv3 = self.conv3(x_down3)
        # x_lk3 = self.elk3(x_down3, 7)
        x_conv3 = replace_feature(x_conv3, self.act3(x_conv3.features + x_lk3.features))
        
        x_down4 = self.down4(x_conv3)
        x_conv4 = self.conv4_tail(self.conv4(x_down4))
        x_lk4 = self.elk4_tail(self.elk4(x_down4, self.block_sz))
        # x_conv4 = self.conv4(x_down4)
        # x_lk4 = self.elk4(x_down4, 7)
        x_conv4 = replace_feature(x_conv4, self.act4(x_conv4.features + x_lk4.features))
        
        ret = self.extra_conv(x_conv4)

        ret = ret.dense()

        N, C, D, H, W = ret.shape
        ret = ret.view(N, C * D, H, W)

        multi_scale_voxel_features = {
            'conv1': x_conv1,
            'conv2': x_conv2,
            'conv3': x_conv3,
            'conv4': x_conv4,
        }

        return ret, multi_scale_voxel_features

