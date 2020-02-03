import os
import cv2
import lmdb
import torch
import pickle
import numpy as np
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from config import get_config
from torch.utils.data import Dataset, DataLoader


IMG_EXTENSIONS = ['jpg', 'jpeg', 'png']


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def _get_paths_from_images(path):
    assert os.path.isdir(path), '{:s} is not a valid directory'.format(path)
    images = []
    for dirpath, _, fnames in sorted(os.walk(path)):
        for fname in sorted(fnames):
            if is_image_file(fname):
                img_path = os.path.join(dirpath, fname)
                images.append(img_path)
    assert images, '{:s} has no valid image file'.format(path)
    return images


def _get_paths_from_lmdb(dataroot):
    env = lmdb.open(dataroot, readonly=True, lock=False, readahead=False, meminit=False)
    keys_cache_file = os.path.join(dataroot, '_keys_cache.p')

    if os.path.isfile(keys_cache_file):
        keys = pickle.load(open(keys_cache_file, "rb"))
    else:
        with env.begin(write=False) as txn:
            keys = [key.decode('ascii') for key, _ in txn.cursor()]
        pickle.dump(keys, open(keys_cache_file, 'wb'))
    paths = sorted([key for key in keys if not key.endswith('.meta')])
    return env, paths


def get_image_paths(data_type, dataroot):
    env, paths = None, None
    if dataroot is not None:
        if data_type == 'lmdb':
            env, paths = _get_paths_from_lmdb(dataroot)
        elif data_type == 'img':
            paths = sorted(_get_paths_from_images(dataroot))
        else:
            raise NotImplementedError('data_type [{:s}] is not recognized.'.format(data_type))
    return env, paths


def _read_lmdb_img(env, path):
    with env.begin(write=False) as txn:
        buf = txn.get(path.encode('ascii'))
        buf_meta = txn.get((path + '.meta').encode('ascii')).decode('ascii')
    img_flat = np.frombuffer(buf, dtype=np.uint8)
    H, W, C = [int(s) for s in buf_meta.split(',')]
    img = img_flat.reshape(H, W, C)
    return img


def read_img(env, path):
    # read image by cv2 or from lmdb
    # return: Numpy float32, HWC, BGR, [0,1]
    if env is None:  # img
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    else:
        img = _read_lmdb_img(env, path)
    img = img.astype(np.float32) / 255.
    if img.ndim == 2:
        img = np.expand_dims(img, axis=2)
    # some images have 4 channels
    if img.shape[2] > 3:
        img = img[:, :, :3]
    return img


class ImageDataset(Dataset):
    '''Read images'''

    def __init__(self, path):
        super(ImageDataset, self).__init__()
        self.paths = None
        self.env = None  # environment for lmdb

        # read image list from lmdb or image files
        self.env, self.paths = get_image_paths('lmdb', path)
        assert self.paths, 'Error: paths are empty.'

    def __getitem__(self, index):
        path = self.paths[index]
        img = read_img(self.env, path)

        # BGR to RGB, HWC to CHW, numpy to tensor, [0, 1] to [-1, 1]
        img = img[:, :, [2, 1, 0]]
        img = torch.from_numpy(np.ascontiguousarray(np.transpose(img, (2, 0, 1)))).float()
        img = img * 2.0 - 1.0

        return img, torch.zeros(0)

    def __len__(self):
        return len(self.paths)

def load_dataset(path):
    cfg = get_config()
    transform = [
        transforms.ToTensor(),
        transforms.Normalize(
            [0.5 for _ in range(cfg['DATASET']['N_CHANNELS'])],
            [0.5 for _ in range(cfg['DATASET']['N_CHANNELS'])]
        )
    ]

    if cfg['DATASET']['NAME'] == 'MNIST':
        transform.insert(0, transforms.Resize([32, 32]))
        transform = transforms.Compose(transform)
        train_data = datasets.MNIST(path, train=True, download=True, transform=transform)
    elif cfg['DATASET']['NAME'] == 'CIFAR10':
        transform = transforms.Compose(transform)
        train_data = datasets.CIFAR10(path, train=True, download=True, transform=transform)
    elif cfg['DATASET']['NAME'] == 'CELEBA64':
        train_data = ImageDataset(path)
    else:
        assert False, 'You can use dataset that is [MNIST, CIFAR10, CELEBA64]'

    train_loader = DataLoader(
        train_data,
        batch_size=cfg['TRAINING']['BATCH_SIZE'],
        shuffle=True,
        num_workers=4,
    )

    return train_data, train_loader