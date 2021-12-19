import logging
import os

import nibabel
import numpy
import pandas
from torchvision.datasets import VisionDataset

logger = logging.getLogger()
FILEPATHKEY = "9dof_2mm_vol"


class UKBBBrainAGE(VisionDataset):
    @staticmethod
    def get_path(root, path):
        if path == "/" or root is None:
            return path
        return os.path.join(root, str(path))

    def __init__(self, root, metadatafile, transform=None, target_transform=None, verify=False,
                 num_sample=-1,  random_state=0):
        super().__init__(root, transform=transform, target_transform=target_transform)
        self.df = pandas.read_csv(metadatafile)

        # do a random sample of dataset
        if num_sample > 0:
            # fixed seed will be useful to train multiple models with same data
            self.df = self.df.sample(n=num_sample, random_state=random_state, replace=True)

        if verify:
            # remove all those entries for which we dont have file
            indices = []
            for i, row in self.df.iterrows():
                if not os.path.exists(self.get_path(root, row[FILEPATHKEY])):
                    indices.append(i)
            if indices:
                logger.info(f"Dropping {len(indices)}")
                logger.debug(f"Dropped rows {indices}")
            self.df = self.df.drop(index=indices)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        path = self.get_path(self.root, row[FILEPATHKEY])
        subject_id = row["subject_id"]
        age = row["age_at_scan"]
        img = nibabel.load(path).get_fdata()
        img = (img - img.mean()) / img.std()
        scan = img[numpy.newaxis, :, :, :]
        age = age

        if self.transform:
            scan = self.transform(scan)

        if self.target_transform:
            age = self.target_transform(age)

        return numpy.float32(scan), numpy.float32(age), subject_id

    def __len__(self):
        return self.df.shape[0]