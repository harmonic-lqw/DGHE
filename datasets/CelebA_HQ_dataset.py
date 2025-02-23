from torch.utils.data import Dataset
import lmdb
from io import BytesIO
from PIL import Image
import torchvision.transforms as tfs
import os

from glob import glob
import shutil

class MultiResolutionDataset(Dataset):
    def __init__(self, path, transform, resolution=256):
        self.env = lmdb.open(
            path,
            max_readers=32,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )

        if not self.env:
            raise IOError("Cannot open lmdb dataset", path)

        with self.env.begin(write=False) as txn:
            self.length = int(txn.get("length".encode("utf-8")).decode("utf-8"))

        self.resolution = resolution
        self.transform = transform

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        with self.env.begin(write=False) as txn:
            key = f"{self.resolution}-{str(index).zfill(5)}".encode("utf-8")
            img_bytes = txn.get(key)

        buffer = BytesIO(img_bytes)
        img = Image.open(buffer)
        img = self.transform(img)

        return img


def get_celeba_dataset(data_root, config):
    train_transform = tfs.Compose([tfs.ToTensor(),
                                   tfs.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5),
                                                 inplace=True)])

    test_transform = tfs.Compose([tfs.ToTensor(),
                                  tfs.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5),
                                                inplace=True)])

    train_dataset = MultiResolutionDataset(os.path.join(data_root, 'LMDB_train'),
                                           train_transform, config.data.image_size)
    test_dataset = MultiResolutionDataset(os.path.join(data_root, 'LMDB_test'),
                                          test_transform, config.data.image_size)


    return train_dataset, test_dataset

################################################################################

class CelebA_dataset(Dataset):
    def __init__(self, image_root, transform=None, mode='train', img_size=256):
        super().__init__()
        self.image_paths = glob(os.path.join(image_root, mode, '*.jpg'))
        # if mode == 'test':
        #     print("="*30)
        #     print(self.image_paths[:300])
        #     destination_folder = '/HDDdata/LQW/Dataset/CelebAHQ/test_300'
        #     for i, img in enumerate(self.image_paths[:300]):
        #         destination_path = os.path.join(destination_folder, f"{i:03d}.jpg")
        #         shutil.copyfile(img, destination_path)
        #     print("="*30)
        self.transform = transform
        self.img_size = img_size

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        x = Image.open(image_path)
        x = x.resize((self.img_size, self.img_size))
        if self.transform is not None:
            x = self.transform(x)
        return x

    def __len__(self):
        return len(self.image_paths)

def get_celeba_dataset2(data_root, config):
    train_transform = tfs.Compose([tfs.ToTensor(),
                                   tfs.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5),
                                                 inplace=True)])

    test_transform = tfs.Compose([tfs.ToTensor(),
                                  tfs.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5),
                                                inplace=True)])

    train_dataset = CelebA_dataset(data_root, transform=train_transform, mode='train',
                                 img_size=config.data.image_size)
    test_dataset = CelebA_dataset(data_root, transform=test_transform, mode='test',
                                img_size=config.data.image_size)


    return train_dataset, test_dataset


