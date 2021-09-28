from pathlib import Path
from typing import Union, List, Optional

import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torchvision import transforms

from src.datamodules.hisDBDataModule.cropped_hisdb_dataset import CroppedHisDBDataset
from src.datamodules.hisDBDataModule.image_folder_segmentation import ImageFolderSegmentationDataset
from src.datamodules.hisDBDataModule.util.analytics.image_analytics import get_analytics
from src.datamodules.hisDBDataModule.util.misc import validate_path
from src.datamodules.hisDBDataModule.util.transformations import transforms as custom_transforms
from src.datamodules.hisDBDataModule.util.transformations.transforms import TwinRandomCrop, OnlyTarget, OnlyImage
from src.utils import template_utils

log = template_utils.get_logger(__name__)


class DIVAHisDBDataModuleCropped(pl.LightningDataModule):
    def __init__(self, data_dir: str = None,
                 selection_train: Optional[Union[int, List[str]]] = None,
                 selection_val: Optional[Union[int, List[str]]] = None,
                 selection_test: Optional[Union[int, List[str]]] = None,
                 crop_size: int = 256, num_workers: int = 4, batch_size: int = 8,
                 shuffle: bool = True, drop_last_batch: bool = True):
        super().__init__()

        analytics = get_analytics(input_path=Path(data_dir),
                                  get_gt_data_paths_func=CroppedHisDBDataset.get_gt_data_paths)

        self.mean = analytics['mean']
        self.std = analytics['std']
        self.class_encodings = analytics['class_encodings']
        self.num_classes = len(self.class_encodings)
        self.class_weights = analytics['class_weights']

        self.image_transform = OnlyImage(transforms.Compose([transforms.ToTensor(),
                                                             transforms.Normalize(mean=self.mean, std=self.std)]))
        self.target_transform = OnlyTarget(transforms.Compose([
            # transforms the gt image into a one-hot encoded matrix
            custom_transforms.OneHotEncodingDIVAHisDB(class_encodings=self.class_encodings),
            # transforms the one hot encoding to argmax labels -> for the cross-entropy criterion
            custom_transforms.OneHotToPixelLabelling()]))
        self.twin_transform = TwinRandomCrop(crop_size=crop_size)

        self.num_workers = num_workers
        self.batch_size = batch_size

        self.shuffle = shuffle
        self.drop_last_batch = drop_last_batch

        self.data_dir = validate_path(data_dir)

        self.selection_train = selection_train
        self.selection_val = selection_val
        self.selection_test = selection_test

        self.dims = (3, crop_size, crop_size)

    def setup(self, stage: Optional[str] = None):
        if stage == 'fit' or stage is None:
            self.train = CroppedHisDBDataset(**self._create_dataset_parameters('train'), selection=self.selection_train)
            self.val = CroppedHisDBDataset(**self._create_dataset_parameters('val'), selection=self.selection_val)

            self._check_min_num_samples(num_samples=len(self.train), data_split='train',
                                        drop_last_batch=self.drop_last_batch)
            self._check_min_num_samples(num_samples=len(self.val), data_split='val',
                                        drop_last_batch=self.drop_last_batch)

        if stage == 'test' or stage is not None:
            self.test = CroppedHisDBDataset(**self._create_dataset_parameters('test'), selection=self.selection_test)
            # self._check_min_num_samples(num_samples=len(self.test), data_split='test',
            #                             drop_last_batch=False)

    def _check_min_num_samples(self, num_samples: int, data_split: str, drop_last_batch: bool):
        num_processes = self.trainer.num_processes
        batch_size = self.batch_size
        if drop_last_batch:
            if num_samples < (self.trainer.num_processes * self.batch_size):
                log.error(
                    f'#samples ({num_samples}) in "{data_split}" smaller than '
                    f'#processes({num_processes}) times batch size ({batch_size}). '
                    f'This only works if drop_last_batch is false!')
                raise ValueError()
        else:
            if num_samples < (self.trainer.num_processes * self.batch_size):
                log.warning(
                    f'#samples ({num_samples}) in "{data_split}" smaller than '
                    f'#processes ({num_processes}) times batch size ({batch_size}). '
                    f'This works due to drop_last_batch=False, however samples will occur multiple times. '
                    f'Check if this behavior is intended!')

    def train_dataloader(self, *args, **kwargs) -> DataLoader:
        return DataLoader(self.train,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          shuffle=self.shuffle,
                          drop_last=self.drop_last_batch,
                          pin_memory=True)

    def val_dataloader(self, *args, **kwargs) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(self.val,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          shuffle=self.shuffle,
                          drop_last=self.drop_last_batch,
                          pin_memory=True)

    def test_dataloader(self, *args, **kwargs) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(self.test,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          shuffle=False,
                          drop_last=False,
                          pin_memory=True)

    def _create_dataset_parameters(self, dataset_type: str = 'train'):
        is_test = dataset_type == 'test'
        return {'path': self.data_dir / dataset_type,
                'image_transform': self.image_transform,
                'target_transform': self.target_transform,
                'twin_transform': self.twin_transform,
                'classes': self.class_encodings,
                'is_test': is_test}

    def get_img_name_coordinates(self, index):
        """
        Returns the original filename of the crop and its coordinate based on the index.
        You can just use this during testing!
        :param index:
        :return:
        """
        if not hasattr(self, 'test'):
            raise Exception('This method can just be called during testing')

        return self.test.img_paths_per_page[index][2:]


class DIVAHisDBDataModuleCB55(pl.LightningDataModule):

    def __init__(self, data_dir: str = None, crop_size: int = 256, num_workers: int = 4, imgs_in_memory: int = 4,
                 crops_per_image: int = 50, batch_size: int = 8):
        super().__init__()

        analytics = get_analytics(input_path=Path(data_dir),
                                  get_gt_data_paths_func=ImageFolderSegmentationDataset.get_gt_data_paths)

        self.mean = analytics['mean']
        self.std = analytics['std']
        self.class_encodings = analytics['class_encodings']
        self.num_classes = len(self.class_encodings)
        self.class_weights = analytics['class_weights']

        # self.mean = [0.38432901678928616, 0.338196317524483, 0.2947592254146755]
        # self.std = [0.4301543541910016, 0.4091020577292445, 0.35576108643619914]
        # self.class_encodings = [1.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0]
        # self.num_classes = len(self.class_encodings)
        #
        # self.class_weights = [6.728128705623194e-08, 6.476751128196739e-07, 2.70544094104861e-05,
        #                       0.00022974558342931251,
        #                       5.840629267581712e-07, 0.004876526983176354, 5.386943668098171e-05, 0.9948115045679762]

        self.transforms = OnlyImage(transforms.Compose([transforms.ToTensor(),
                                                        transforms.Normalize(
                                                            mean=self.mean,
                                                            std=self.std),
                                                        transforms.ToPILImage()]))
        self.target_transform = OnlyTarget(transforms.Compose([
            # transforms the gt image into a one-hot encoded matrix
            custom_transforms.OneHotEncodingDIVAHisDB(class_encodings=self.class_encodings),
            # transforms the one hot encoding to argmax labels -> for the cross-entropy criterion
            custom_transforms.OneHotToPixelLabelling()]))
        self.twin_transform = TwinRandomCrop(crop_size=crop_size)

        self.crop_size = crop_size
        self.num_workers = num_workers
        self.imgs_in_memory = imgs_in_memory
        self.crops_per_image = crops_per_image
        self.batch_size = batch_size

        self.data_dir = validate_path(data_dir)

    def setup(self, stage: Optional[str] = None):
        if stage == 'fit' or stage is None:
            self.his_train = ImageFolderSegmentationDataset(**self._create_dataset_parameters('train'))
            self.his_val = ImageFolderSegmentationDataset(**self._create_dataset_parameters('val'))

        if stage == 'test' or stage is not None:
            self.his_test = ImageFolderSegmentationDataset(**self._create_dataset_parameters('test'))

    def train_dataloader(self, *args, **kwargs) -> DataLoader:
        return DataLoader(self.his_train,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          pin_memory=True)

    def val_dataloader(self, *args, **kwargs) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(self.his_val,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          pin_memory=True)

    def test_dataloader(self, *args, **kwargs) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(self.his_test,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          pin_memory=True)

    def _create_dataset_parameters(self, dataset_type: str = 'train'):
        is_test = dataset_type == 'test'
        return {'path': self.data_dir / dataset_type,
                'workers': self.num_workers,  # 'workers': 1 if is_test else self.num_workers,
                'imgs_in_memory': self.imgs_in_memory,
                'crops_per_image': self.crops_per_image,
                'crop_size': self.crop_size,
                'transform': self.transforms,
                'target_transform': self.target_transform,
                'twin_transform': None if is_test else self.twin_transform,
                'classes': self.class_encodings,
                'is_test': is_test}


if __name__ == '__main__':
    data = DIVAHisDBDataModuleCB55('tests/dummy_data/dummy_dataset', num_workers=5)

    from tqdm import tqdm

    data.setup('test')
    for (batch, i) in tqdm(data.test_dataloader()):
        print(i)
        print(batch)
