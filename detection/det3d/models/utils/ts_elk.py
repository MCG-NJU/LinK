import torch
import torch.nn as nn
from torchsparse import SparseTensor
import spconv.pytorch as spconv
from torch.utils.checkpoint import checkpoint
import torchsparse.nn as spnn



def spconv2ts(sct: spconv.SparseConvTensor):
    """
        sct: SparseConvTensor (spconv)
        
        returns: 
          - st: SparseTensor (torchsparse)
          - sct_save: a dict including some info from sct,
                     for the use of inverse transformation.
    """
    feats = sct.features
    coords = torch.index_select(sct.indices, 1, 
             torch.LongTensor([3,2,1,0]).to(sct.indices.device)).contiguous()
    st = SparseTensor(feats, coords, 1)

    sct_save = dict()
    sct_save['batch_size'] = sct.batch_size
    sct_save['benchmark'] = sct.benchmark
    sct_save['benchmark_record'] = sct.benchmark_record
    sct_save['grid'] = sct.grid
    sct_save['indice_dict'] = sct.indice_dict
    sct_save['spatial_shape'] = sct.spatial_shape
    sct_save['voxel_num'] = sct.voxel_num

    return st, sct_save
    

def ts2spconv(st: SparseTensor, sct_save: dict):
    """
        - st: SparseTensor (torchsparse)
        - sct_save: a dict including some additional info for sct
                    
        returns: 
            sct: SparseConvTensor (spconv)
    """
    features = st.feats
    indices = torch.index_select(st.coords, 1, 
              torch.LongTensor([3,2,1,0]).to(st.coords.device)).contiguous()
    sct = spconv.SparseConvTensor(
                features,
                indices,
                spatial_shape=sct_save['spatial_shape'],
                batch_size=sct_save['batch_size'],
                grid=sct_save['grid'],
                voxel_num=sct_save['voxel_num'],
                indice_dict=sct_save['indice_dict'],
                benchmark=sct_save['benchmark']
    )
    sct.benchmark_record = sct_save['benchmark_record']

    return sct


import torchsparse.nn.functional as F
from torchsparse.nn.utils import get_kernel_offsets


# x: SparseTensor->large, stride: scale of kernel 
# return : SparseTensor->small
def large_to_small(large_x, stride):
    x_C = torch.cat([torch.div(large_x.C[:,:3], stride, rounding_mode='floor').int(), large_x.C[:,3:]], dim=1)
    large_x_hash = F.sphash(x_C.to(large_x.F.device))
    small_x_C = torch.unique(x_C, dim=0)
    small_x_hash = F.sphash(small_x_C.to(large_x.F.device))

    idx_query = F.sphashquery(large_x_hash, small_x_hash)
    counts = F.spcount(idx_query.int(), len(small_x_hash))
    inserted_feat = F.spvoxelize(large_x.F, idx_query, counts)
    small_x = SparseTensor(inserted_feat, small_x_C, stride)

    small_x.cmaps = large_x.cmaps
    small_x.kmaps = large_x.kmaps
    return small_x, idx_query, counts


def small_to_large_v2(small_x, large_x, idx, counts):
    # local offsets to index neighbors
    ## [2^3,3]
    kernel_size = 3
    offsets = get_kernel_offsets(kernel_size, 1, 1, device=large_x.F.device)
    neighbor_hash = F.sphash(
        small_x.C, offsets
    )

    small_hash = F.sphash(small_x.C.to(large_x.F.device))

    idx_query = F.sphashquery(neighbor_hash, small_hash)
    idx_query = idx_query.transpose(0,1).contiguous()
    idx_query_flat = idx_query.view(-1)
    f = torch.cat([small_x.F, torch.ones_like(small_x.F[:,:1]).to(small_x.F.device)], dim=1)
    f = f*counts.unsqueeze(dim=-1)
    weights = torch.ones(small_x.F.shape[0], kernel_size**3).to(small_x.F.device).float()
    weights[idx_query == -1] = 0
    new_feat = F.spdevoxelize(f, idx_query, weights, kernel_size)
    new_feat = new_feat[:,:-1] / new_feat[:,-1:]

    large_x.F = new_feat[idx]

    return large_x


class TSELKBlock(nn.Module):
    def __init__(self, inc, outc, baseop='cos'):
        super().__init__()
        self.inc = inc
        self.outc = outc
        self.baseop = baseop
        self.pre_mix = nn.Sequential(
            nn.Linear(self.inc, self.inc, bias = False),
            nn.LayerNorm(self.inc, eps=1e-6)
        )
        self.local_mix = nn.Sequential(
            spnn.Conv3d(self.inc, self.inc, kernel_size=3, dilation=1,
                            stride=1),
        )

        self.pos_weight = nn.Sequential(
            nn.Linear(3, self.inc, bias=False),
        )
        
        self.norm = nn.LayerNorm(self.inc, eps=1e-6)
        self.norm_local = nn.LayerNorm(self.inc, eps=1e-6)
        self.activate = nn.ReLU(True)


    def forward(self, sct: spconv.SparseConvTensor, stride):
        """
            sct: SparseConvTensor (in spconv)
        """
        st, sct_save = spconv2ts(sct)
        new_st = self.forward_(st, stride)
        new_sct = ts2spconv(new_st, sct_save)
        return new_sct


    def forward_(self, st: SparseTensor, stride):
        '''
            st: SparseTensor
            stride: scale of large kernel
        '''
        F_input = self.pre_mix(st.F)
        local_mix = self.local_mix(st)
        
        if self.baseop == 'sin':
            pos_weight = self.pos_weight(st.C[:,:3].float())
            pos_weight_sin = torch.sin(pos_weight)
            pos_weight_cos = torch.cos(pos_weight)
            F_weighted_sin = F_input*pos_weight_sin
            F_weighted_cos = F_input*pos_weight_cos
            st.F = torch.cat([F_weighted_sin, F_weighted_cos], dim=1).contiguous()

            small_st, idx, counts = large_to_small(st, stride=stride)
            large_st = small_to_large_v2(small_st, st, idx, counts)


            new_st_F = large_st.F[:,:self.inc]*pos_weight_cos - large_st.F[:,self.inc:]*pos_weight_sin

        elif self.baseop == 'cos':
            pos_weight = self.pos_weight(st.C[:,:3].float())
            pos_weight = pos_weight[:,:self.inc//2].repeat([1,2]) # channel grouping
            pos_weight_sin = torch.sin(pos_weight)
            pos_weight_cos = torch.cos(pos_weight)
            F_weighted_sin = F_input*pos_weight_sin
            F_weighted_cos = F_input*pos_weight_cos
            st.F = torch.cat([F_weighted_cos, F_weighted_sin], dim=1).contiguous()

            small_st, idx, counts = large_to_small(st, stride=stride)
            large_st = small_to_large_v2(small_st, st, idx, counts)

            new_st_F = large_st.F[:,:self.inc]*pos_weight_cos + large_st.F[:,self.inc:]*pos_weight_sin

        elif self.baseop == 'cos_x_alpha':
            pos_weight = self.pos_weight(st.C[:,:3].float())*self.alpha
            pos_weight = pos_weight[:,:self.inc//2].repeat([1,2]) # channel grouping

            pos_weight_sin = torch.sin(pos_weight)
            pos_weight_cos = torch.cos(pos_weight)
            F_weighted_sin = F_input*pos_weight_sin
            F_weighted_cos = F_input*pos_weight_cos
            F_weighted_linear = F_input*pos_weight
            st.F = torch.cat([F_weighted_cos, F_weighted_sin, F_weighted_linear], dim=1).contiguous()

            small_st, idx, counts = large_to_small(st, stride=stride)
            large_st = small_to_large_v2(small_st, st, idx, counts)

            new_st_F = large_st.F[:,:self.inc]*pos_weight_cos + large_st.F[:,self.inc:2*self.inc]*pos_weight_sin + (large_st.F[:,2*self.inc:]-F_weighted_linear)

        elif self.baseop == 'cos_sin':
            pos_weight = self.pos_weight(st.C[:,:3].float())

            pos_weight_sin = torch.sin(pos_weight)
            pos_weight_cos = torch.cos(pos_weight)
            F_weighted_sin = F_input*pos_weight_sin
            F_weighted_cos = F_input*pos_weight_cos
            st.F = torch.cat([F_weighted_cos, F_weighted_sin], dim=1).contiguous()

            small_st, idx, counts = large_to_small(st, stride=stride)
            large_st = small_to_large_v2(small_st, st, idx, counts)

            new_st_F = (large_st.F[:,:self.inc]*pos_weight_cos + large_st.F[:,self.inc:]*pos_weight_sin) + \
            (large_st.F[:,self.inc:]*pos_weight_cos - large_st.F[:,:self.inc]*pos_weight_sin)

        elif self.baseop == 'x':
            pos_weight = self.pos_weight(st.C[:,:3].float())
            pos_weight = pos_weight[:,:self.inc//2].repeat([1,2])
            
            F_weighted_linear = F_input*pos_weight
            st.F = torch.cat([F_weighted_linear], dim=1).contiguous()

            small_st, idx, counts = large_to_small(st, stride=stride)
            large_st = small_to_large_v2(small_st, st, idx, counts)


            new_st_F = large_st.F-F_weighted_linear

        new_st_F = self.norm(new_st_F)
        local_F = self.norm_local(local_mix.F)
        new_st_F = self.activate(new_st_F+local_F)

        large_st.F = new_st_F

        return large_st


# no additional bn in v2
# CAUTION: DEPRECATED! Used in v2!
class TSELKBlock_no_tail_norm(nn.Module):
    def __init__(self, inc, outc, baseop='cos'):
        super().__init__()
        self.inc = inc
        self.outc = outc
        self.baseop = baseop
        self.pre_mix = nn.Sequential(
            nn.Linear(self.inc, self.inc, bias = False),
            nn.LayerNorm(self.inc, eps=1e-6)
        )
        self.pos_weight = nn.Sequential(
            nn.Linear(3, self.inc, bias=False),
        )
        self.norm = nn.BatchNorm1d(self.inc) 


    def forward(self, sct: spconv.SparseConvTensor, stride):
        """
            sct: SparseConvTensor (in spconv)
        """
        st, sct_save = spconv2ts(sct)
        new_st = self.forward_(st, stride)
        new_sct = ts2spconv(new_st, sct_save)
        return new_sct


    def forward_(self, st: SparseTensor, stride):
        '''
            st: SparseTensor
            stride: scale of large kernel
        '''
        # 1. pos weight
        # F_input = st.F # self.pre_mix(st.F)
        F_input = self.pre_mix(st.F)
        if self.baseop == 'exp':
            pos_weight = torch.exp(self.pos_weight(st.C[:,:3].float()) / 100.0)
            F_weighted = F_input*pos_weight
            st.F = F_weighted

            small_st, idx = large_to_small(st, stride=stride)
            large_st = small_to_large(small_st, st, idx)

            new_st_F = large_st.F / pos_weight
        elif self.baseop == 'sin':
            pos_weight = self.pos_weight(st.C[:,:3].float())
            pos_weight_sin = torch.sin(pos_weight)
            pos_weight_cos = torch.cos(pos_weight)
            F_weighted_sin = F_input*pos_weight_sin
            F_weighted_cos = F_input*pos_weight_cos
            st.F = torch.cat([F_weighted_sin, F_weighted_cos], dim=1).contiguous()

            small_st, idx, counts = large_to_small(st, stride=stride)
            large_st = small_to_large_v2(small_st, st, idx, counts)


            new_st_F = large_st.F[:,:self.inc]*pos_weight_cos - large_st.F[:,self.inc:]*pos_weight_sin

        elif self.baseop == 'cos':
            pos_weight = self.pos_weight(st.C[:,:3].float())
            pos_weight_sin = torch.sin(pos_weight)
            pos_weight_cos = torch.cos(pos_weight)
            F_weighted_sin = F_input*pos_weight_sin
            F_weighted_cos = F_input*pos_weight_cos
            st.F = torch.cat([F_weighted_cos, F_weighted_sin], dim=1).contiguous()

            small_st, idx, counts = large_to_small(st, stride=stride)
            large_st = small_to_large_v2(small_st, st, idx, counts)

            new_st_F = large_st.F[:,:self.inc]*pos_weight_cos + large_st.F[:,self.inc:]*pos_weight_sin

        # new_st_F = self.norm(new_st_F)

        large_st.F = new_st_F

        return large_st

