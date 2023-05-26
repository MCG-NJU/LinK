import torch
import torchsparse.nn.functional as F
from torchsparse import PointTensor, SparseTensor
from torchsparse.nn.utils import get_kernel_offsets
from torchsparse.utils import make_ntuple
import numpy as np

import time


def ravel_hash(x: np.ndarray) -> np.ndarray:
    assert x.ndim == 2, x.shape

    x = x - np.min(x, axis=0)
    x = x.astype(np.uint64, copy=False)
    xmax = np.max(x, axis=0).astype(np.uint64) + 1

    h = np.zeros(x.shape[0], dtype=np.uint64)
    for k in range(x.shape[1] - 1):
        h += x[:, k]
        h *= xmax[k + 1]
    h += x[:, -1]
    return h


def vox_sample(pt: PointTensor, voxel_size):
    '''Return indices and inverse_indices'''
    xyz = pt.C[:, :-1]
    strided_coords = torch.cat([
        torch.floor(xyz / voxel_size).int() * voxel_size,
        pt.C[:, -1].int().view(-1, 1)
        ], 1)

    pc_hash = ravel_hash(strided_coords.cpu().numpy())
    _, indices, inverse_indices = np.unique(pc_hash,
                                            return_index=True,
                                            return_inverse=True)

    return indices, inverse_indices


# x: SparseTensor: voxel, stride: scale of kernel 
# return : SparseTensor: aux
def voxel_to_aux(large_x, s):
    x_C = torch.cat([torch.div(large_x.C[:,:3], s, rounding_mode='floor').int(), large_x.C[:,3:]], dim=1)
    large_x_hash = F.sphash(x_C.to(large_x.F.device))
    small_x_C = torch.unique(x_C, dim=0)
    small_x_hash = F.sphash(small_x_C.to(large_x.F.device))

    idx_query = F.sphashquery(large_x_hash, small_x_hash)
    counts = F.spcount(idx_query.int(), len(small_x_hash))
    inserted_feat = F.spvoxelize(large_x.F, idx_query, counts)
    small_x = SparseTensor(inserted_feat, small_x_C, s)

    small_x.cmaps = large_x.cmaps
    small_x.kmaps = large_x.kmaps
    
    return small_x, idx_query, counts


def aux_to_voxel(small_x, large_x, idx, counts, r=2):
    # local offsets to index neighbors
    ## [2^3,3]
    # kernel_size = 2
    offsets = get_kernel_offsets(r, 1, 1, device=large_x.F.device)
    neighbor_hash = F.sphash(
        small_x.C, offsets
    )

    small_hash = F.sphash(small_x.C.to(large_x.F.device))

    idx_query = F.sphashquery(neighbor_hash, small_hash)
    idx_query = idx_query.transpose(0,1).contiguous()
    idx_query_flat = idx_query.view(-1)
    f = torch.cat([small_x.F, torch.ones_like(small_x.F[:,:1]).to(small_x.F.device)], dim=1)
    f = f*counts.unsqueeze(dim=-1)
    weights = torch.ones(small_x.F.shape[0], r**3).to(small_x.F.device).float()
    weights[idx_query == -1] = 0
    new_feat = F.spdevoxelize(f, idx_query, weights, r)
    new_feat = new_feat[:,:-1] / new_feat[:,-1:]

    large_x.F = new_feat[idx]

    return large_x

def get_idx_counts(st: SparseTensor, pt: PointTensor):
    if pt.additional_features is None or pt.additional_features.get(
            'idx_query') is None or pt.additional_features['idx_query'].get(
                (st.s, st.shift)) is None:
        print("INFO: calculating...")
        stride = st.s

        pc_coords = torch.cat([
            torch.floor(pt.C[:, :3] / stride).int() * stride,
            pt.C[:, -1].int().view(-1, 1)
            ], 1)
        pc_hash = F.sphash(pc_coords)
        sparse_hash = pc_hash.unique()
        
        idx_query = F.sphashquery(pc_hash, sparse_hash)
        counts = F.spcount(idx_query.int(), st.C.shape[0])
        pt.additional_features['idx_query'][(st.s, st.shift)] = idx_query
        pt.additional_features['counts'][(st.s, st.shift)] = counts    
    else:
        idx_query = pt.additional_features['idx_query'][(st.s, st.shift)]
        counts = pt.additional_features['counts'][(st.s, st.shift)]
    
    return idx_query, counts


def calc_ti_weights(coords: torch.Tensor,
                    idx_query: torch.Tensor,
                    scale: float = 1) -> torch.Tensor:
    with torch.no_grad():
        p = coords
        if scale != 1:
            pf = torch.floor(coords / scale) * scale
        else:
            pf = torch.floor(coords)
        pc = pf + scale

        x = p[:, 0].view(-1, 1)
        y = p[:, 1].view(-1, 1)
        z = p[:, 2].view(-1, 1)

        xf = pf[:, 0].view(-1, 1).float()
        yf = pf[:, 1].view(-1, 1).float()
        zf = pf[:, 2].view(-1, 1).float()

        xc = pc[:, 0].view(-1, 1).float()
        yc = pc[:, 1].view(-1, 1).float()
        zc = pc[:, 2].view(-1, 1).float()

        w0 = (xc - x) * (yc - y) * (zc - z)
        w1 = (xc - x) * (yc - y) * (z - zf)
        w2 = (xc - x) * (y - yf) * (zc - z)
        w3 = (xc - x) * (y - yf) * (z - zf)
        w4 = (x - xf) * (yc - y) * (zc - z)
        w5 = (x - xf) * (yc - y) * (z - zf)
        w6 = (x - xf) * (y - yf) * (zc - z)
        w7 = (x - xf) * (y - yf) * (z - zf)

        w = torch.cat([w0, w1, w2, w3, w4, w5, w6, w7], dim=1)
        w = w.transpose(1, 0).contiguous()
        if scale != 1:
            w /= scale ** 3
        w[idx_query == -1] = 0
        w /= torch.sum(w, dim=0) + 1e-8
    return w




# Below are not used for now.

def zero_rows_frac(feat: torch.Tensor):
    '''feat is a 2d tensor'''
    summed = feat.sum(dim=1)
    nonzero_rows = summed.count_nonzero()
    frac = 1 - (nonzero_rows / feat.shape[0]).item()
    return frac


def neg_one_frac(idx_query: torch.Tensor):
    return len(idx_query[idx_query==-1]) / len(idx_query)


def tensor_stat(t: torch.Tensor):
    print(f"{t.shape} {t.min():>10.3f}{t.max():>10.3f}{t.float().mean():>10.3f}")


def splitBN(tsr: torch.Tensor, num_batches: int, mode: str) -> torch.Tensor:
    '''mode 'C': convert a tensor from shape (BN, 4) to (B, N, 3)
       mode 'F': convert a tensor from shape (BN, C) to (B, N, C)
    '''
    assert mode in ['C', 'F']
    num_points = tsr.shape[0] // num_batches
    if mode == 'C':
        out = torch.zeros(num_batches, num_points, 3).cuda()
        for i in range(num_batches):
            out[i, ...] = tsr[i*num_points:(i+1)*num_points, :3]
    elif mode == 'F':
        num_channels = tsr.shape[1]
        out = torch.zeros(num_batches, num_points, num_channels).cuda()
        for i in range(num_batches):
            out[i, ...] = tsr[i*num_points:(i+1)*num_points, :]
    return out


def mergeBN(tsr: torch.Tensor, mode: str) -> torch.Tensor:
    '''mode 'C': convert a tensor from shape (B, N, 3) to (BN, 4)
       mode 'F': convert a tensor from shape (B, N, C) to (BN, C)
    '''
    num_batches = tsr.shape[0]
    num_points = tsr.shape[1]
    num_channels = tsr.shape[2]
    if mode == 'C':
        out = torch.zeros(num_batches * num_points, 4).cuda()
        for i in range(num_batches):
            out[i*num_points:(i+1)*num_points, 0] = i
            out[i*num_points:(i+1)*num_points, :3] = tsr[i, ...]
    elif mode == 'F':
        out = torch.zeros(num_batches * num_points, num_channels).cuda()
        for i in range(num_batches):
            out[i*num_points:(i+1)*num_points, :] = tsr[i, ...]
    return out


def vector_gather(vectors, indices):
    """
    Gathers (batched) vectors according to indices.
    Arguments:
        vectors: Tensor[N, L, D]
        indices: Tensor[N, K] or Tensor[N]
    Returns:
        Tensor[N, K, D] or Tensor[N, D]
    """
    N, L, D = vectors.shape
    squeeze = False
    if indices.ndim == 1:
        squeeze = True
        indices = indices.unsqueeze(-1)
    N2, K = indices.shape
    assert N == N2
    indices = einops.repeat(indices, "N K -> N K D", D=D)
    out = torch.gather(vectors, dim=1, index=indices)
    if squeeze:
        out = out.squeeze(1)
    return out


# z: PointTensor
# return: SparseTensor
def initial_voxelize(z, init_res, after_res):
    new_float_coord = torch.cat(
        [(z.C[:, :3] * init_res) / after_res, z.C[:, -1].view(-1, 1)], 1)

    pc_hash = F.sphash(torch.floor(new_float_coord).int())
    sparse_hash = torch.unique(pc_hash)
    idx_query = F.sphashquery(pc_hash, sparse_hash)
    counts = F.spcount(idx_query.int(), len(sparse_hash))

    inserted_coords = F.spvoxelize(torch.floor(new_float_coord), idx_query,
                                   counts)
    inserted_coords = torch.round(inserted_coords).int()
    inserted_feat = F.spvoxelize(z.F, idx_query, counts)

    new_tensor = SparseTensor(inserted_feat, inserted_coords, 1)
    new_tensor.cmaps.setdefault(new_tensor.stride, new_tensor.coords)
    z.additional_features['idx_query'][1] = idx_query
    z.additional_features['counts'][1] = counts
    z.C = new_float_coord

    return new_tensor


# x: SparseTensor, z: PointTensor
# return: SparseTensor
def point_to_voxel(x, z):
    if z.additional_features is None or z.additional_features.get(
            'idx_query') is None or z.additional_features['idx_query'].get(
                x.s) is None:
        pc_hash = F.sphash(
            torch.cat([
                torch.floor(z.C[:, :3] / x.s[0]).int() * x.s[0],
                z.C[:, -1].int().view(-1, 1)
            ], 1))
        sparse_hash = F.sphash(x.C)
        idx_query = F.sphashquery(pc_hash, sparse_hash)
        counts = F.spcount(idx_query.int(), x.C.shape[0])
        z.additional_features['idx_query'][x.s] = idx_query
        z.additional_features['counts'][x.s] = counts
    else:
        idx_query = z.additional_features['idx_query'][x.s]
        counts = z.additional_features['counts'][x.s]

    inserted_feat = F.spvoxelize(z.F, idx_query, counts)
    new_tensor = SparseTensor(inserted_feat, x.C, x.s)
    new_tensor.cmaps = x.cmaps
    new_tensor.kmaps = x.kmaps

    return new_tensor


# x: SparseTensor, z: PointTensor
# return: PointTensor
def voxel_to_point(x, z, nearest=False):
    if z.idx_query is None or z.weights is None or z.idx_query.get(
            x.s) is None or z.weights.get(x.s) is None:
        off = get_kernel_offsets(2, x.s, 1, device=z.F.device)
        old_hash = F.sphash(
            torch.cat([
                torch.floor(z.C[:, :3] / x.s[0]).int() * x.s[0],
                z.C[:, -1].int().view(-1, 1)
            ], 1), off)
        pc_hash = F.sphash(x.C.to(z.F.device))
        idx_query = F.sphashquery(old_hash, pc_hash)
        weights = F.calc_ti_weights(z.C, idx_query,
                                    scale=x.s[0]).transpose(0, 1).contiguous()
        idx_query = idx_query.transpose(0, 1).contiguous()
        if nearest:
            weights[:, 1:] = 0.
            idx_query[:, 1:] = -1
        new_feat = F.spdevoxelize(x.F, idx_query, weights)
        new_tensor = PointTensor(new_feat,
                                 z.C,
                                 idx_query=z.idx_query,
                                 weights=z.weights)
        new_tensor.additional_features = z.additional_features
        new_tensor.idx_query[x.s] = idx_query
        new_tensor.weights[x.s] = weights
        z.idx_query[x.s] = idx_query
        z.weights[x.s] = weights

    else:
        new_feat = F.spdevoxelize(x.F, z.idx_query.get(x.s), z.weights.get(x.s))
        new_tensor = PointTensor(new_feat,
                                 z.C,
                                 idx_query=z.idx_query,
                                 weights=z.weights)
        new_tensor.additional_features = z.additional_features

    return new_tensor

# x: SparseTensor, coarse scale; ref_x: SparseTensor, fine scale
# return: SparseTensor
def upsample_voxel(x, ref_x):
    stride = x.s[0]
    x_C = torch.cat([torch.div(x.C[:,:3], stride, rounding_mode='floor').int(), x.C[:,3:]], dim=1)
    ref_x_C = torch.cat([torch.div(ref_x.C[:,:3], stride, rounding_mode='floor').int(), ref_x.C[:,3:]], dim=1)
    
    x_hash = F.sphash(x_C.to(x.F.device))
    ref_x_hash = F.sphash(ref_x_C.to(x.F.device))
    idx_query = F.sphashquery(ref_x_hash, x_hash) # 
    queried_feat = x.F[idx_query]

    new_tensor = SparseTensor(queried_feat, ref_x.C, ref_x.s)
    new_tensor.cmaps.setdefault(new_tensor.stride, new_tensor.coords)

    return new_tensor

if __name__ == '__main__':
    N = 10

    feats = torch.rand(N, 4).cuda()
    coords = torch.rand(N, 4).cuda()
    coords[:,:3] *= 3
    coords[:,3] = 0
    coords = coords.round()
    pt = PointTensor(feats=feats, coords=coords)
    st = initial_voxelize(pt, 1, 1)
    aux_st, idx, counts = voxel_to_aux(st, 2)
    voxel_st = aux_to_voxel(aux_st, st, idx, counts)
