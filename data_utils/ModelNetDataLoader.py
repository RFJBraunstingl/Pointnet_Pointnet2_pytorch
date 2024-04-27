'''
@author: Xu Yan
@file: ModelNet.py
@time: 2021/3/19 15:51
'''
import os
import numpy as np
import warnings
import pickle
import random
import torch

from tqdm import tqdm
from torch.utils.data import Dataset

warnings.filterwarnings('ignore')


def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc


def farthest_point_sample(point, npoint):
    """
    Input:
        xyz: pointcloud data, [N, D]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [npoint, D]
    """
    N, D = point.shape
    xyz = point[:,:3]
    centroids = np.zeros((npoint,))
    distance = np.ones((N,)) * 1e10
    farthest = np.random.randint(0, N)
    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = np.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, -1)
    point = point[centroids.astype(np.int32)]
    return point


def map_to_synth_path(path_to_model_file):
    # i.e.: 'bottle_0001.txt' -> 'bottle_001.txt'
    # i.e.: 'airplane_0660.txt' -> 'airplane_660.txt'
    # i.e.: 'tv_stand_0123.txt' -> 'tv_stand_123.txt'
    parts = path_to_model_file.split('_')
    parts[-1] = parts[-1][1:]
    result = '_'.join(parts)
    return result


def apply_elastic_distortion_augmentation(point_set, granularity=[0.1], magnitude=[0.2]):
    device = point_set.device
    cur_type = point_set.dtype
    coords = point_set.detach().clone()

    blurx = torch.ones((3, 1, 3, 1, 1)).to(cur_type).to(device) / 3
    blury = torch.ones((3, 1, 1, 3, 1)).to(cur_type).to(device) / 3
    blurz = torch.ones((3, 1, 1, 1, 3)).to(cur_type).to(device) / 3

    coords_min = torch.amin(coords, 0).reshape((1, -1))
    coords_max = torch.amax(coords, 0).reshape((1, -1))
    noise_dims_full = torch.amax(coords - coords_min, 0)

    for cur_granularity, cur_magnitude in zip(granularity, magnitude):

        noise_dim = (noise_dims_full // cur_granularity).to(torch.int32) + 3
        noise = torch.randn(1, 3, *noise_dim).to(cur_type).to(device)

        # Smoothing.
        convolve = torch.nn.functional.conv3d
        for _ in range(2):
            noise = convolve(noise, blurx, padding='same', groups=3)
            noise = convolve(noise, blury, padding='same', groups=3)
            noise = convolve(noise, blurz, padding='same', groups=3)

        # Trilinear interpolate noise filters for each spatial dimensions.
        sample_coords = ((coords - coords_min)/(coords_max - coords_min))*2. - 1.
        sample_coords = sample_coords.reshape(1, -1, 1, 1, 3) # [N, 1, 1, 3]
        new_sample_coords = sample_coords.clone()
        new_sample_coords[..., 0] = sample_coords[..., 2]
        new_sample_coords[..., 2] = sample_coords[..., 0]
        sample = torch.nn.functional.grid_sample(
            noise, new_sample_coords, align_corners=True,
            padding_mode='border')[0,:,:,0,0].transpose(0,1)

        coords += sample * cur_magnitude

    return coords


class ModelNetDataLoader(Dataset):
    def __init__(self, root, args, split='train', process_data=False):
        self.root = root
        self.npoints = args.num_point
        self.process_data = process_data
        self.uniform = args.use_uniform_sample
        self.use_normals = args.use_normals
        self.num_category = args.num_category
        self.random_choice_sampling = args.use_random_choice_sampling
        self.synthetic_augmentation_probability = args.synthetic_augmentation_probability if split == 'train' else 0
        self.noise_augmentation_probability = args.noise_augmentation_probability if split == 'train' else 0
        self.noise_augmentation_stddev = args.noise_augmentation_stddev
        self.rotation_augmentation_probability = args.rotation_augmentation_probability if split == 'train' else 0
        self.distortion_augmentation_probability = args.distortion_augmentation_probability if split == 'train' else 0

        shape_names_path = f"modelnet{self.num_category}_shape_names.txt"
        train_path = f"modelnet{self.num_category}_train.txt"
        test_path = f"modelnet{self.num_category}_test.txt"

        self.catfile = os.path.join(self.root, shape_names_path)
        self.cat = [line.rstrip() for line in open(self.catfile)]
        self.classes = dict(zip(self.cat, range(len(self.cat))))

        shape_ids = {}
        shape_ids['train'] = [line.rstrip() for line in open(os.path.join(self.root, train_path))]
        shape_ids['test'] = [line.rstrip() for line in open(os.path.join(self.root, test_path))]

        assert (split == 'train' or split == 'test')
        shape_names = ['_'.join(x.split('_')[0:-1]) for x in shape_ids[split]]
        self.datapath = [(shape_names[i], os.path.join(self.root, shape_names[i], shape_ids[split][i]) + '.txt') for i
                         in range(len(shape_ids[split]))]
        print('The size of %s data is %d' % (split, len(self.datapath)))

        if self.uniform:
            self.save_path = os.path.join(root, 'modelnet%d_%s_%dpts_fps.dat' % (self.num_category, split, self.npoints))
        else:
            self.save_path = os.path.join(root, 'modelnet%d_%s_%dpts.dat' % (self.num_category, split, self.npoints))

        if self.process_data:
            if not os.path.exists(self.save_path):
                print('Processing data %s (only running in the first time)...' % self.save_path)
                self.list_of_points = [None] * len(self.datapath)
                self.list_of_labels = [None] * len(self.datapath)

                for index in tqdm(range(len(self.datapath)), total=len(self.datapath)):
                    fn = self.datapath[index]
                    cls = self.classes[self.datapath[index][0]]
                    cls = np.array([cls]).astype(np.int32)
                    point_set = np.loadtxt(fn[1], delimiter=',').astype(np.float32)

                    if self.uniform:
                        point_set = farthest_point_sample(point_set, self.npoints)
                    else:
                        point_set = point_set[0:self.npoints, :]

                    self.list_of_points[index] = point_set
                    self.list_of_labels[index] = cls

                with open(self.save_path, 'wb') as f:
                    pickle.dump([self.list_of_points, self.list_of_labels], f)
            else:
                print('Load processed data from %s...' % self.save_path)
                with open(self.save_path, 'rb') as f:
                    self.list_of_points, self.list_of_labels = pickle.load(f)

    def __len__(self):
        return len(self.datapath)

    def _get_item(self, index):
        if self.process_data:
            point_set, label = self.list_of_points[index], self.list_of_labels[index]
        else:
            fn = self.datapath[index]
            path_to_model_file = fn[1]

            if random.random() < self.synthetic_augmentation_probability:
                path_to_model_file = map_to_synth_path(path_to_model_file)

            cls = self.classes[self.datapath[index][0]]
            label = np.array([cls]).astype(np.int32)
            point_set = np.loadtxt(path_to_model_file, delimiter=',').astype(np.float32)
            num_of_points_in_set = len(point_set)

            if self.uniform:
                point_set = farthest_point_sample(point_set, self.npoints)
            elif self.random_choice_sampling:
                random_indices = np.random.choice(range(num_of_points_in_set), self.npoints)
                point_set = point_set[random_indices, :]
            else:
                point_set = point_set[0:self.npoints, :]

            if random.random() < self.noise_augmentation_stddev:
                noise = np.random.normal(0,self.noise_augmentation_stddev,point_set.shape)
                # noise = torch.randn(point_set.shape) * self.noise_augmentation_stddev
                point_set = point_set + noise

            if random.random() < self.rotation_augmentation_probability:
                pass
                #min_angle = 0
                #max_angle = 2*np.pi
                #cur_angle = torch.rand(1).item() * (max_angle - min_angle) + min_angle
                #r = torch.from_numpy(
                #    np.array([[1.0, 0.0, 0.0],
                #              [0.0, np.cos(cur_angle), -np.sin(cur_angle)],
                #              [0.0, np.sin(cur_angle), np.cos(cur_angle)]])).to(device).to(torch.float32)
                #point_set = torch.matmul(point_set, r)

            if random.random() < self.distortion_augmentation_probability:
                point_set = apply_elastic_distortion_augmentation(point_set)

        point_set[:, 0:3] = pc_normalize(point_set[:, 0:3])
        if not self.use_normals:
            point_set = point_set[:, 0:3]

        return point_set, label[0]

    def __getitem__(self, index):
        return self._get_item(index)


if __name__ == '__main__':
    import torch

    data = ModelNetDataLoader('/data/modelnet40_normal_resampled/', split='train')
    DataLoader = torch.utils.data.DataLoader(data, batch_size=12, shuffle=True)
    for point, label in DataLoader:
        print(point.shape)
        print(label.shape)
