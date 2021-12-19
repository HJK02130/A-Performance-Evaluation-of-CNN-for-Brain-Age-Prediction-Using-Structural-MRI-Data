import logging

from box import Box
from torchvision.transforms import transforms

from src.common.data.ukbb_brain_age import UKBBBrainAGE
from src.common.data_utils import frame_drop

logger = logging.getLogger()


def get_dataset(name, test_csv=None, train_csv=None, valid_csv=None, root_path=None,
                train_num_sample=-1, frame_keep_style="random", frame_keep_fraction=1.0,
                frame_dim=1, impute=False, **kwargs):
    """ return dataset """

    if name == "brain_age":
        # Transformations to remove frames
        frame_drop_transform = lambda x: frame_drop(x, frame_keep_style=frame_keep_style,
                                                    frame_keep_fraction=frame_keep_fraction,
                                                    frame_dim=frame_dim, impute=impute)
        # Transformation to add noise to frames
        transform = transforms.Compose([frame_drop_transform])
        train_data = UKBBBrainAGE(root=root_path, metadatafile=train_csv,
                                  num_sample=train_num_sample, transform=transform)
        test_data = UKBBBrainAGE(root=root_path, metadatafile=test_csv, transform=transform)
        valid_data = UKBBBrainAGE(root=root_path, metadatafile=valid_csv if valid_csv else test_csv,
                                  transform=transform)
        return Box({"train": train_data, "test": test_data, "valid": valid_data}), {}

    logger.error(f"Invalid data name {name} specified")
    raise Exception(f"Invalid data name {name} specified")
