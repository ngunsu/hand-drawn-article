from pytorch_lightning import LightningDataModule
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset


class FondefWorkpiecesHoles(Dataset):

    def __init__(self,
                 dataset_root: str = '/datasets/fondef_id20I10262/workpieces_holes/',
                 split_type: str = 'k4_1',
                 split_partition: str = 'train') -> None:
        """
        Parameters
        ----------
        dataset_root : str
            Absolute path of workpieces dataset
        split_type : str
            How to divide the data [full_data|full_train|k4_1|...|k4_4]
        split_partition : str
            Requiered split partition [train|val|test]
        Returns
        -------
        None
        """
        super().__init__()
        self.dataset_root = dataset_root
        self.split_type = split_type
        self.split_partition = split_partition
        self.load_dataset()

    def load_dataset(self) -> None:
        """
        Loads workpieces holes dataset. Images are RGB of size 128x128. GT X,Y relative position between [0, 1]
        """
        # 0 - 255 images
        self.images = np.load(f'{self.dataset_root}/{self.split_type}/{self.split_partition}_im.npy')
        self.gt = np.load(f'{self.dataset_root}/{self.split_type}/{self.split_partition}_gt.npy')
        # Convert to tensor
        self.gt = torch.tensor(self.gt, dtype=torch.float32)
        self.images = torch.tensor(self.images/255, dtype=torch.float32)
        self.images = self.images.permute(0, 3, 1, 2)

    def __len__(self):
        """Returns the number of images in the dataset"""
        return self.gt.shape[0]

    def __getitem__(self, idx: int):
        """
        Returns an image and  a GT file

        Parameters
        ----------
        idx : int
            Requested image ID

        Returns
        -------
        tuple
            (image, GT) -> GT = {}
        """
        im = self.images[idx]
        gt = self.gt[idx]

        return im, gt


class FondefWorkpiecesPointDataModule(LightningDataModule):

    def __init__(self,
                 split_type='k4_1',
                 batch_size=64,
                 im_size=(128, 128),
                 shuffle=True,
                 num_workers=20):
        super().__init__()
        self.split_type = split_type
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.im_size = im_size

    def prepare_data(self):
        dataset = FondefWorkpiecesHoles

        self.train = dataset(split_type=self.split_type,
                             split_partition='train')
        self.val = dataset(split_type=self.split_type,
                           split_partition='val')
        self.test = dataset(split_type='test',
                            split_partition='test')

        if self.split_type == 'full_data' or self.split_type == 'full_train':
            self.val = self.train
            self.test = self.test

    def train_dataloader(self):
        return DataLoader(self.train,
                          batch_size=self.batch_size,
                          shuffle=self.shuffle,
                          num_workers=self.num_workers,
                          drop_last=True)

    def val_dataloader(self):
        return DataLoader(self.val,
                          batch_size=self.batch_size,
                          shuffle=False,
                          num_workers=self.num_workers,
                          drop_last=False)

    def test_dataloader(self):
        return DataLoader(self.test,
                          batch_size=1,
                          shuffle=False,
                          num_workers=self.num_workers,
                          drop_last=False)
