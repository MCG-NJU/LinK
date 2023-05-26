import imp
import os
import os.path
import glob
import random

import numpy as np
from torchsparse import SparseTensor
from torchsparse.utils.collate import sparse_collate_fn
from torchsparse.utils.quantize import sparse_quantize


__all__ = ['SemanticKITTI']

label_name_mapping = {
    0: 'unlabeled',         ## 0, unlabeled
    1: 'outlier',           ## 0
    10: 'car',              # 1
    11: 'bicycle',          # 2
    13: 'bus',              ## 5, other vehicle
    15: 'motorcycle',       # 3
    16: 'on-rails',         ## 5, other vehicle
    18: 'truck',            # 4
    20: 'other-vehicle',    # 5
    30: 'person',           # 6
    31: 'bicyclist',        # 7
    32: 'motorcyclist',     # 8
    40: 'road',             # 9
    44: 'parking',          # 10
    48: 'sidewalk',         # 11
    49: 'other-ground',     # 12
    50: 'building',         # 13
    51: 'fence',            # 14
    52: 'other-structure',  ## 0, unlabeled
    60: 'lane-marking',     ## 9, road
    70: 'vegetation',       # 15
    71: 'trunk',            # 16
    72: 'terrain',          # 17
    80: 'pole',             # 18
    81: 'traffic-sign',     # 19
    99: 'other-object',     ## 0, unlabeled
    252: 'moving-car',      ## 1, car
    253: 'moving-bicyclist', ## 7, bicyclist
    254: 'moving-person',   ## 8, person
    255: 'moving-motorcyclist', ## 9, motorcyclist
    256: 'moving-on-rails', ## 5, other vehicle
    257: 'moving-bus',      ## 5, other vehicle
    258: 'moving-truck',    ## 4, truck
    259: 'moving-other-vehicle' ## 5, other vehicle
}

learning_mapping = {
  0 : 0,     # "unlabeled"
  1 : 0,     # "outlier" mapped to "unlabeled" --------------------------mapped
  10: 1,     # "car"
  11: 2,     # "bicycle"
  13: 5,     # "bus" mapped to "other-vehicle" --------------------------mapped
  15: 3,     # "motorcycle"
  16: 5,     # "on-rails" mapped to "other-vehicle" ---------------------mapped
  18: 4,     # "truck"
  20: 5,     # "other-vehicle"
  30: 6,     # "person"
  31: 7,     # "bicyclist"
  32: 8,     # "motorcyclist"
  40: 9,     # "road"
  44: 10,    # "parking"
  48: 11,    # "sidewalk"
  49: 12,    # "other-ground"
  50: 13,    # "building"
  51: 14,    # "fence"
  52: 0,     # "other-structure" mapped to "unlabeled" ------------------mapped
  60: 9,     # "lane-marking" to "road" ---------------------------------mapped
  70: 15,    # "vegetation"
  71: 16,    # "trunk"
  72: 17,    # "terrain"
  80: 18,    # "pole"
  81: 19,    # "traffic-sign"
  99: 0,     # "other-object" to "unlabeled" ----------------------------mapped
  252: 1,    # "moving-car" to "car" ------------------------------------mapped
  253: 7,    # "moving-bicyclist" to "bicyclist" ------------------------mapped
  254: 6,    # "moving-person" to "person" ------------------------------mapped
  255: 8,    # "moving-motorcyclist" to "motorcyclist" ------------------mapped
  256: 5,    # "moving-on-rails" mapped to "other-vehicle" --------------mapped
  257: 5,    # "moving-bus" mapped to "other-vehicle" -------------------mapped
  258: 4,    # "moving-truck" to "truck" --------------------------------mapped
  259: 5,    # "moving-other"-vehicle to "other-vehicle" ----------------mapped 

}


# adjust the order to match the index of outputs
kept_labels = [
    'unlabeled', 'car', 'bicycle', 'motorcycle', 'truck', 'other-vehicle', 'person',
    'bicyclist', 'motorcyclist', 'road', 'parking', 'sidewalk', 'other-ground',
    'building', 'fence', 'vegetation', 'trunk', 'terrain', 'pole', 'traffic-sign'
]


class SemanticKITTI(dict):

    def __init__(self, root, voxel_size, num_points, use_tta=False, finetune=False, **kwargs):
        sample_stride = kwargs.get('sample_stride', 1)
        self.use_tta = use_tta
        super().__init__({
            'train':
                SemanticKITTIInternal(root,
                                        voxel_size,
                                        num_points,
                                        use_tta=False,
                                        sample_stride=1,
                                        split='train',
                                        finetune=finetune),
            'val':
                SemanticKITTIInternal(root,
                                        voxel_size,
                                        num_points,
                                        use_tta=self.use_tta,
                                        sample_stride=sample_stride,
                                        split='val'),
            'test':
                SemanticKITTIInternal(root,
                                        voxel_size,
                                        num_points,
                                        use_tta=self.use_tta,
                                        sample_stride=sample_stride,
                                        split='test')
        
        })


class SemanticKITTIInternal:

    def __init__(self,
                 root,
                 voxel_size,
                 num_points,
                 split,
                 sample_stride=1,
                 use_tta=False,
                 finetune=False
                 ):

        self.root = root
        self.split = split
        self.voxel_size = voxel_size
        self.num_points = num_points
        self.sample_stride = sample_stride
        self.use_tta = use_tta

        self.seqs = []


        # given train/val/test seqs
        if split == 'train':
            self.seqs = [
                '00', '01', '02', '03', '04', '05', '06', '07', '09', '10',
            ]
        elif self.split == 'val':
            self.seqs = ['08']
        elif self.split == 'test':
            self.seqs = [
                '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21'
            ]

        
        self.files = []
        for seq in self.seqs:
            seq_files = sorted(
                os.listdir(os.path.join(self.root, seq, 'velodyne')))
            seq_files = [
                os.path.join(self.root, seq, 'velodyne', x) for x in seq_files
            ]
            self.files.extend(seq_files)

        if self.sample_stride > 1:
            self.files = self.files[::self.sample_stride]

        # reverse_label_name_mapping = {}
        reverse_label_name_mapping = {'unlabeled': 0, 'car': 1, 'bicycle': 2, 'motorcycle': 3, 'truck': 4, 'other-vehicle': 5, 'person': 6,
            'bicyclist': 7, 'motorcyclist': 8, 'road': 9, 'parking': 10, 'sidewalk': 11, 'other-ground': 12,
            'building': 13, 'fence': 14, 'vegetation': 15, 'trunk': 16, 'terrain': 17, 'pole': 18, 'traffic-sign': 19
        }
        self.label_map = np.full(260, 255)
        for i in range(260):
            if i in learning_mapping:
                self.label_map[i] = learning_mapping[i]

        self.reverse_label_name_mapping = reverse_label_name_mapping
        self.num_classes = 20
        self.angle = 0.0


    def set_angle(self, angle):
        self.angle = angle

    def __len__(self):
        return len(self.files)
    
    def get_single_sample(self, block, labels_, index, use_aug=True, num_voting=-1):
        if use_aug:
            ## rot aug
            theta = np.random.uniform(0*np.pi, 2 * np.pi)          
            scale_factor = np.random.uniform(0.95, 1.05)
            rot_mat = np.array([[np.cos(theta), np.sin(theta), 0],
                                [-np.sin(theta),
                                    np.cos(theta), 0], [0, 0, 1]])       
            block[:, :3] = np.dot(block[:, :3], rot_mat) * scale_factor 

            ## flip aug
             
            flip_type = np.random.choice(4, 1)
            if flip_type == 1:
                block[:, 0] = -block[:, 0]
            elif flip_type == 2:
                block[:, 1] = -block[:, 1]
            elif flip_type == 3:
                block[:, :2] = -block[:, :2]
           
        pc_ = np.round(block[:, :3] / self.voxel_size).astype(np.int32)
        pc_ -= pc_.min(0, keepdims=1)
        
        feat_ = block
        coords, inds, inverse_map = sparse_quantize(pc_,
                                return_index=True,
                                return_inverse=True)

        if 'train' in self.split:
            if len(inds) > self.num_points:
                inds = np.random.choice(inds, self.num_points, replace=False)
        
        if num_voting !=-1:
            pc_ = np.concatenate([pc_, np.ones_like(pc_[:,:1])*num_voting], axis=-1)
            pc = pc_[inds]
            feat = feat_[inds]
            labels = labels_[inds]
            
            return pc, pc_, feat, labels, labels_, inverse_map
        else:
            pc = pc_[inds]
            feat = feat_[inds]
            labels = labels_[inds]
            lidar = SparseTensor(feat, pc)
            labels = SparseTensor(labels, pc)
            labels_ = SparseTensor(labels_, pc_)
            inverse_map = SparseTensor(inverse_map, pc_)


        return (lidar, labels, labels_, inverse_map)

    def __getitem__(self, index):
        with open(self.files[index], 'rb') as b:
            # Raw data is 1d, in shape (4N). Reshape to (N,4)
            block_ = np.fromfile(b, dtype=np.float32).reshape(-1, 4) # (x,y,z,batch)
        block = np.zeros_like(block_)
        block[...] = block_[...]

        if 'test' not in self.split:
            label_file = self.files[index].replace('velodyne', 'labels').replace(
                '.bin', '.label')
            if os.path.exists(label_file):
                with open(label_file, 'rb') as a:
                    all_labels = np.fromfile(a, dtype=np.int32).reshape(-1)
            else:
                all_labels = np.zeros(block.shape[0]).astype(np.int32)
            labels_ = self.label_map[all_labels & 0xFFFF].astype(np.int64)
            # instance_labels = (all_labels >> 16).astype(np.int64)

        else:
            labels_ = np.zeros(block.shape[0]).astype(np.int64)

        
        
        if 'train' in self.split:
            lidar, labels, labels_, inverse_map = self.get_single_sample(block, labels_, index)

            return {
                        'lidar': lidar,
                        'targets': labels,
                        'targets_mapped': labels_,
                        'inverse_map': inverse_map,
                        'file_name': self.files[index]
                    }
        else:
            if not self.use_tta:
                lidar, labels, labels_, inverse_map = self.get_single_sample(block, labels_, index, use_aug=False)
                
                return {
                    'lidar': lidar,
                    'targets': labels,
                    'targets_mapped': labels_,
                    'inverse_map': inverse_map,
                    'file_name': self.files[index]
                }
            else:
                num_voting = 1
                pc_total = []
                pc__total = []
                feat_total = []
                labels_total = []
                labels__total = []
                inverse_map_total = []

                for count in range(num_voting):
                    pc, pc_, feat, labels, labels_, inverse_map = self.get_single_sample(block, labels_, index, use_aug=True, num_voting=count)
                    pc_total.append(pc)
                    pc__total.append(pc_)
                    feat_total.append(feat)
                    labels_total.append(labels)
                    labels__total.append(labels_)
                    inverse_map_total.append(inverse_map)


                pc = np.concatenate(pc_total,  axis=0)
                pc_ = np.concatenate(pc__total,  axis=0)
                feat = np.concatenate(feat_total,  axis=0)
                
                labels = np.concatenate(labels_total,  axis=0)
                labels_ = np.concatenate(labels__total,  axis=0)
                inverse_map = np.concatenate(inverse_map_total,  axis=0)
                
                lidar = SparseTensor(feat, pc)
                labels = SparseTensor(labels, pc)
                labels_ = SparseTensor(labels_, pc_)
                inverse_map = SparseTensor(inverse_map, pc_)

                return {
                    'lidar': lidar,
                    'targets': labels,
                    'targets_mapped': labels_,
                    'inverse_map': inverse_map,
                    'file_name': self.files[index],
                }
         
        return {
            'lidar': lidar,
            'targets': labels,
            'targets_mapped': labels_,
            'inverse_map': inverse_map,
            'file_name': self.files[index]
        }

    @staticmethod
    def collate_fn(inputs):
        return sparse_collate_fn(inputs)
