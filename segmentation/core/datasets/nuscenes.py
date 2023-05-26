import imp
import os
import os.path
import glob
import random
from pathlib import Path

import numpy as np
from torchsparse import SparseTensor
from torchsparse.utils.collate import sparse_collate_fn
from torchsparse.utils.quantize import sparse_quantize
from nuscenes.utils import splits


__all__ = ['nuScenes']

label_name_mapping = {
  0: 'noise',
  1: 'animal',
  2: 'human.pedestrian.adult',
  3: 'human.pedestrian.child',
  4: 'human.pedestrian.construction_worker',
  5: 'human.pedestrian.personal_mobility',
  6: 'human.pedestrian.police_officer',
  7: 'human.pedestrian.stroller',
  8: 'human.pedestrian.wheelchair',
  9: 'movable_object.barrier',
  10: 'movable_object.debris',
  11: 'movable_object.pushable_pullable',
  12: 'movable_object.trafficcone',
  13: 'static_object.bicycle_rack',
  14: 'vehicle.bicycle',
  15: 'vehicle.bus.bendy',
  16: 'vehicle.bus.rigid',
  17: 'vehicle.car',
  18: 'vehicle.construction',
  19: 'vehicle.emergency.ambulance',
  20: 'vehicle.emergency.police',
  21: 'vehicle.motorcycle',
  22: 'vehicle.trailer',
  23: 'vehicle.truck',
  24: 'flat.driveable_surface',
  25: 'flat.other',
  26: 'flat.sidewalk',
  27: 'flat.terrain',
  28: 'static.manmade',
  29: 'static.other',
  30: 'static.vegetation',
  31: 'vehicle.ego',
}

learning_mapping = {
  1: 0,
  5: 0,
  7: 0,
  8: 0,
  10: 0,
  11: 0,
  13: 0,
  19: 0,
  20: 0,
  0: 0,
  29: 0,
  31: 0,
  9: 1,
  14: 2,
  15: 3,
  16: 3,
  17: 4,
  18: 5,
  21: 6,
  2: 7,
  3: 7,
  4: 7,
  6: 7,
  12: 8,
  22: 9,
  23: 10,
  24: 11,
  25: 12,
  26: 13,
  27: 14,
  28: 15,
  30: 16
}

# adjust the order to match the index of outputs
kept_labels =['noise', 'barrier', 'bicycle', 'bus', 'car', 'construction_vehicle', 'motorcycle',
            'pedestrian', 'traffic_cone', 'trailer', 'truck', 'driveable_surface', 'other_flat',
            'sidewalk', 'terrain', 'manmade', 'vegetation']


class nuScenes(dict):

    def __init__(self, root, voxel_size, num_points, use_tta=False, finetune=False, **kwargs):
        sample_stride = kwargs.get('sample_stride', 1)
        self.use_tta = use_tta
        super().__init__({
            'train':
                nuScenesInternal(root,
                                voxel_size,
                                num_points,
                                use_tta=False,
                                sample_stride=1,
                                split='train',
                                ),
            'val':
                nuScenesInternal(root,
                                voxel_size,
                                num_points,
                                use_tta=True,
                                sample_stride=sample_stride,
                                split='val'),
            'test':
                nuScenesInternal(root,
                                voxel_size,
                                num_points,
                                use_tta=True,
                                sample_stride=sample_stride,
                                split='test')
        
        })


class nuScenesInternal:
    def __init__(self,
                 root,
                 voxel_size,
                 num_points,
                 split,
                 sample_stride=1,
                 use_tta=False,
                 ):

        self.root = root
        self.split = split
        self.voxel_size = voxel_size
        self.num_points = num_points
        self.sample_stride = sample_stride
        self.use_tta = use_tta
        self.num_vote = 1
        
        debug = False
        if debug:
            version = 'v1.0-mini'
            scenes = splits.mini_train
        else:
            if split != 'test':
                version = 'v1.0-trainval'
                if split == 'train':
                    scenes = splits.train
                else:
                    scenes = splits.val
            else:
                version = 'v1.0-test'
                scenes = splits.test
                
        from nuscenes import NuScenes
        self.nusc = NuScenes(version=version, dataroot=root, verbose=True)
        self.get_available_scenes()
        available_scene_names = [s['name'] for s in self.available_scenes]
        scenes = list(filter(lambda x: x in available_scene_names, scenes))
        scenes = set([self.available_scenes[available_scene_names.index(s)]['token'] for s in scenes])
        self.get_path_infos_lidar(scenes)
        
        reverse_label_name_mapping = {'noise': 0, 'barrier': 1, 'bicycle': 2, 'bus': 3, 'car': 4, 'construction_vehicle': 5, 'motorcycle': 6,
            'pedestrian': 7, 'traffic_cone': 8, 'trailer': 9, 'truck': 10, 'driveable_surface': 11, 'other_flat': 12,
            'sidewalk': 13, 'terrain': 14, 'manmade': 15, 'vegetation': 16}
        self.label_map = np.full(260, 255)
        for i in range(260):
            if i in learning_mapping:
                self.label_map[i] = learning_mapping[i]

        self.reverse_label_name_mapping = reverse_label_name_mapping
        self.num_classes = 20
        self.angle = 0.0

    def get_available_scenes(self):
        # only for check if all the files are available
        self.available_scenes = []
        for scene in self.nusc.scene:
            scene_token = scene['token']
            scene_rec = self.nusc.get('scene', scene_token)
            sample_rec = self.nusc.get('sample', scene_rec['first_sample_token'])
            sd_rec = self.nusc.get('sample_data', sample_rec['data']['LIDAR_TOP'])
            has_more_frames = True
            scene_not_exist = False
            while has_more_frames:
                lidar_path, _, _ = self.nusc.get_sample_data(sd_rec['token'])
                if not Path(lidar_path).exists():
                    scene_not_exist = True
                    break
                else:
                    break

            if scene_not_exist:
                continue
            self.available_scenes.append(scene)

    def get_path_infos_lidar(self, scenes):
        self.token_list = []

        for sample in self.nusc.sample:
            scene_token = sample['scene_token']
            lidar_token = sample['data']['LIDAR_TOP']  # 360 lidar

            if scene_token in scenes:
                for _ in range(self.num_vote):
                    self.token_list.append(
                        {'lidar_token': lidar_token}
                    )

    def loadDataByIndex(self, index):
        lidar_sample_token = self.token_list[index]['lidar_token']
        lidar_path = os.path.join(self.root,
                                  self.nusc.get('sample_data', lidar_sample_token)['filename'])
        raw_data = np.fromfile(lidar_path, dtype=np.float32).reshape((-1, 5))

        if self.split == 'test':
            self.lidarseg_path = None
            labels_ = np.zeros_like(raw_data[:, 0], dtype=int)
        else:
            lidarseg_path = os.path.join(self.root,
                                         self.nusc.get('lidarseg', lidar_sample_token)['filename'])
            annotated_data = np.fromfile(
                lidarseg_path, dtype=np.uint8).reshape(-1)  # label
            
            labels_ = self.label_map[annotated_data].astype(np.int64)

        pointcloud = raw_data[:, :4]
        sem_label = labels_
        inst_label = np.zeros(pointcloud.shape[0], dtype=np.int32)
        return pointcloud, sem_label, inst_label, lidar_sample_token
    
    def set_angle(self, angle):
        self.angle = angle

    def __len__(self):
        return len(self.token_list)
    
    def get_single_sample(self, block, labels_, use_aug=True, num_voting=-1):
        if use_aug:
            ## rot aug
            theta = np.random.uniform(0*np.pi, 2 * np.pi)          
            scale_factor = np.random.uniform(0.95, 1.05)
            rot_mat = np.array([[np.cos(theta), np.sin(theta), 0],
                                [-np.sin(theta),
                                    np.cos(theta), 0], [0, 0, 1]])       
            block[:, :3] = np.dot(block[:, :3], rot_mat) * scale_factor # np.array([scale_factor, scale_factor, 1.0])

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
        block, labels_, instance_label, lidar_sample_token = self.loadDataByIndex(index)
        
        # data augmentation
        if 'train' in self.split:
            lidar, labels, labels_, inverse_map = self.get_single_sample(block, labels_)

            return {
                        'lidar': lidar,
                        'targets': labels,
                        'targets_mapped': labels_,
                        'inverse_map': inverse_map,
                        # 'file_name': self.files[index]
                    }
        else:
            if not self.use_tta:
                lidar, labels, labels_, inverse_map = self.get_single_sample(block, labels_, use_aug=False)
                
                return {
                    'lidar': lidar,
                    'targets': labels,
                    'targets_mapped': labels_,
                    'inverse_map': inverse_map,
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
                    pc, pc_, feat, labels, labels_, inverse_map = self.get_single_sample(block, labels_, use_aug=True, num_voting=count)
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
                    # 'file_name': self.files[index],
                }
        
        return {
            'lidar': lidar,
            'targets': labels,
            'targets_mapped': labels_,
            'inverse_map': inverse_map,
        }

    @staticmethod
    def collate_fn(inputs):
        return sparse_collate_fn(inputs)
