import copy
import glob
import logging

import torch
import torch.cuda
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import transforms

from dataloader.process_meta_data import get_lesion_infos

log = logging.getLogger(__name__)
# log.setLevel(logging.WARN)
# log.setLevel(logging.INFO)
log.setLevel(logging.DEBUG)


class SkinLesionDataset(Dataset):
    def __init__(self,
                 val_stride=0,
                 is_validation=None,
                 lesion_id=None,
                 transform=None,
                 ):
        if transform:
            self.transform = transform
        else:
            self.transform = transforms.ToTensor()

        self.lesion_infos = copy.copy(get_lesion_infos())

        self.jpgs_paths = glob.glob('/usr/src/rawdata/*.jpg')

        if lesion_id:
            self.lesion_infos = [
                x for x in self.lesion_infos if x.lesion_id == lesion_id
            ]

        if is_validation:
            assert val_stride > 0, val_stride
            self.lesion_infos = self.lesion_infos[::val_stride]
            assert self.lesion_infos
        elif val_stride > 0:
            del self.lesion_infos[::val_stride]
            assert self.lesion_infos

        log.info('{!r}: {} {} samples'.format(
            self,
            len(self.lesion_infos),
            'validation' if is_validation else 'training',
        ))

    def __len__(self):
        return len(self.lesion_infos)

    def __getitem__(self, idx):
        lesion_infos_tup = self.lesion_infos[0][idx]
        image_path = ''
        for path in self.jpgs_paths:
            image_id = path.split('/')[-1].split('.')[0]
            if image_id == lesion_infos_tup.image_id:
                image_path = path
                break

        if len(image_path) == 0:
            log.error('{!r}: {} was not found in dataset'.format(
                self,
                lesion_infos_tup.image_id)
            )

        image = Image.open(image_path).convert('RGB')
        tensor_image = self.transform(image)
        return tensor_image, torch.tensor(lesion_infos_tup[2:]),
