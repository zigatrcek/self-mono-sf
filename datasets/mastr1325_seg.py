from __future__ import absolute_import, division, print_function

import os.path
import torch
import torch.utils.data as data
import numpy as np

from torchvision import transforms as vision_transforms
from .common import read_image_as_byte, read_modd2_calib_into_dict, read_annotation
from .common import kitti_crop_image_list, kitti_adjust_intrinsic
import logging


class MaSTr1325_Base(data.Dataset):
    def __init__(self,
                 args,
                 images_root=None,
                 flip_augmentations=False,
                 preprocessing_crop=True,
                 crop_size=[384, 512],
                 num_examples=-1,
                 index_file=None):

        self._args = args
        self._seq_len = 1
        self._flip_augmentations = flip_augmentations
        self._preprocessing_crop = preprocessing_crop
        self._crop_size = crop_size

        path_dir = os.path.dirname(os.path.realpath(__file__))
        path_index_file = os.path.join(path_dir, index_file)

        # log index file
        logging.info(f'Index file: {path_index_file}')
        logging.info(f'Images root: {images_root}')


        if not os.path.exists(path_index_file):
            raise ValueError("Index File '%s' not found!", path_index_file)
        index_file = open(path_index_file, 'r')

        ## loading image -----------------------------------
        if not os.path.isdir(images_root):
            raise ValueError(f"Image directory '{images_root}' not found!")

        # annotations_root = os.path.normpath(os.path.join(images_root, '../MaSTr1325_masks_512x384'))
        annotations_root = os.path.normpath(os.path.join(images_root, '../masks'))
        logging.info(f'Annotations root: {annotations_root}')


        if not os.path.isdir(annotations_root):
            raise ValueError(f"Annotations directory '{annotations_root}' not found!")

        filename_list = [line.rstrip().split('.')[0] for line in index_file.readlines()]
        logging.info(f'Number of images: {len(filename_list)}')
        self._image_list = []
        self._annotations_list = []
        ext = '.jpg'
        ann_ext = '.png'

        for item in filename_list:
            # parse file name into id, several different options
            if len(item) == 4:
                # only id, 0001
                idx_src = f'{int(item):04}'
                idx_tgt = f'{(int(item) + 1):04}'
                filename_src = idx_src + ext
                filename_tgt = idx_tgt + ext
            elif len(item) == 8:
                # old_ + id, old_0001
                logging.debug(f'item: {item}')
                logging.debug(f'item[-4:]: {item[-4:]}')
                idx_src = f'{int(item[-4:]):04}'
                idx_tgt = f'{(int(item[-4:]) + 1):04}'
                filename_src = 'old_' + idx_src + ext
                filename_tgt = 'old_' + idx_tgt + ext
            else:
                # old_ + 8 digit id + L, old_00000001L
                logging.debug(f'item: {item}')
                logging.debug(f'item[-9:-1]: {item[-9:-1]}')
                idx_src = f'{int(item[-9:-1]):04}'
                idx_tgt = f'{(int(item[-9:-1]) + 1):04}'
                filename_src = 'old_' + idx_src + 'L' + ext
                filename_tgt = 'old_' + idx_tgt + 'L' + ext
            filename_ann_src = filename_src.split('.')[0] + 'm' + ann_ext
            filename_ann_tgt = filename_tgt.split('.')[0] + 'm' + ann_ext

            logging.debug(f'filename_src: {filename_src}')
            name_l1 = os.path.join(images_root, filename_src)
            name_l2 = os.path.join(images_root, filename_tgt)
            logging.debug(f'name_l1: {name_l1}, exists: {os.path.isfile(name_l1)}')
            logging.debug(f'name_l2: {name_l2}, exists: {os.path.isfile(name_l2)}')
            ann_l1 = os.path.join(annotations_root, filename_ann_src)
            ann_l2 = os.path.join(annotations_root, filename_ann_tgt)
            logging.debug(f'ann_l1: {ann_l1}, exists: {os.path.isfile(ann_l1)}')
            logging.debug(f'ann_l2: {ann_l2}, exists: {os.path.isfile(ann_l2)}')

            if all([
                os.path.isfile(name_l1),
                os.path.isfile(name_l2),
                os.path.isfile(ann_l1),
                os.path.isfile(ann_l2)
            ]):
                logging.debug(f'All files exist.')
                self._image_list.append([name_l1, name_l2])
                self._annotations_list.append([ann_l1, ann_l2])
            else:
                logging.debug(f'Not all files exist.')
        logging.info(f'Number of images: {len(self._image_list)}')

        if num_examples > 0:
            # TODO shuffling doesn't do anything since the data is only loaded in the first epoch.
            # logging.info(f'Using {num_examples} examples, shuffling...')
            # np.random.shuffle(self._image_list)
            self._image_list = self._image_list[:num_examples]

        self._size = len(self._image_list)
        logging.info(f'Number of examples: {self._size}')

        ## loading calibration matrix
        self.intrinsic_dict_l = {}
        self.intrinsic_dict_r = {}
        self.intrinsic_dict_l, self.intrinsic_dict_r = read_modd2_calib_into_dict(path_dir)

        self._to_tensor = vision_transforms.Compose([
            vision_transforms.ToPILImage(),
            vision_transforms.transforms.ToTensor()
        ])

    def __getitem__(self, index):
        logging.debug(f'index: {index}')
        logging.debug(f'self._size: {self._size}')
        index = index % self._size

        # read images
        # im_l1, im_l2
        img_list_np = [read_image_as_byte(img) for img in self._image_list[index]]
        # logging.info(f'img_list_np[0].shape: {img_list_np[0].shape}')

        # example filename
        im_l1_filename = self._image_list[index][0]
        im_l2_filename = self._image_list[index][1].split('/')[-1].split('.')[0]
        logging.debug(f'im_l2_filename: {im_l2_filename}')
        # name of sequence directory
        sequence = os.path.basename(os.path.dirname(os.path.dirname(im_l1_filename)))
        # use intrinsic matrix of training data, since we eval on monocular data
        k_l1 = torch.from_numpy(next(iter(self.intrinsic_dict_l.values()))).float()
        k_r1 = torch.from_numpy(next(iter(self.intrinsic_dict_r.values()))).float()

        # input size
        h_orig, w_orig, _ = img_list_np[0].shape
        input_im_size = torch.from_numpy(np.array([h_orig, w_orig])).float()

        # read annotations
        # ann_l1, ann_l2
        logging.debug(f'Loading annotations: {self._annotations_list[index]}')
        ann_list_np = [read_image_as_byte(ann) for ann in self._annotations_list[index]]
        logging.debug(f'ann_list_np[0].shape: {ann_list_np[0].shape}')

        # example filename
        ann_l1_filename = self._annotations_list[index][0]
        ann_l2_filename = self._annotations_list[index][1].split('/')[-1].split('.')[0]
        logging.debug(f'ann_l2_filename: {ann_l2_filename}')
        # name of sequence directory
        sequence = os.path.basename(os.path.dirname(os.path.dirname(ann_l1_filename)))


        # cropping
        if self._preprocessing_crop:

            # get starting positions
            crop_height = self._crop_size[0]
            crop_width = self._crop_size[1]
            x = np.random.uniform(0, w_orig - crop_width + 1)
            y = np.random.uniform(0, h_orig - crop_height + 1)
            crop_info = [int(x), int(y), int(x + crop_width), int(y + crop_height)]

            # cropping images and adjust intrinsic accordingly
            img_list_np = kitti_crop_image_list(img_list_np, crop_info)
            ann_list_np = kitti_crop_image_list(ann_list_np, crop_info)
            k_l1, k_r1 = kitti_adjust_intrinsic(k_l1, k_r1, crop_info)

        # to tensors
        img_list_tensor = [self._to_tensor(img) for img in img_list_np]
        ann_list_tensor = [(self._to_tensor(ann) * 255).long() for ann in ann_list_np]

        im_l1 = img_list_tensor[0]
        im_l2 = img_list_tensor[1]
        ann_l1 = ann_list_tensor[0]
        ann_l2 = ann_list_tensor[1]

        common_dict = {
            'sequence': sequence,
            'img_name': im_l2_filename, # take second image as name
            "index": index,
            "basename": f'{sequence}-{index}',
            "datename": sequence,
            "input_size": input_im_size
        }
        logging.debug(f'common_dict: {common_dict}')

        # random flip
        if self._flip_augmentations is True and torch.rand(1) > 0.5:
            raise ValueError('Flip augmentation not implemented for segmentation dataset.')

        else:
            example_dict = {
                "input_l1": im_l1,
                "input_l2": im_l2,
                "ann_l1": ann_l1,
                "ann_l2": ann_l2,
                "input_k_l1": k_l1,
                "input_k_r1": k_r1,
                "input_k_l2": k_l1,
                "input_k_r2": k_r1,
            }
            example_dict.update(common_dict)
        # logging.info(f'example_dict ann_l1: {example_dict["ann_l1"].shape}')
        # logging.info(f'example_dict ann_l2: {example_dict["ann_l2"].shape}')
        # logging.info(f'example_dict input_l1: {example_dict["input_l1"].shape}')
        # logging.info(f'example_dict input_l2: {example_dict["input_l2"].shape}')
        # logging.info(f'example_dict input_k_l1: {example_dict["input_k_l1"].shape}')
        # logging.info(f'example_dict input_k_r1: {example_dict["input_k_r1"].shape}')
        # logging.info(f'example_dict input_k_l2: {example_dict["input_k_l2"].shape}')
        # logging.info(f'example_dict input_k_r2: {example_dict["input_k_r2"].shape}')
        # exit()
        return example_dict

    def __len__(self):
        return self._size


class MaSTr1325_Full(MaSTr1325_Base):
    def __init__(self,
                 args,
                 root,
                 flip_augmentations=False,
                 preprocessing_crop=True,
                 crop_size=[384, 512],
                 num_examples=-1):
        super(MaSTr1325_Full, self).__init__(
            args,
            images_root=root,
            flip_augmentations=flip_augmentations,
            preprocessing_crop=preprocessing_crop,
            crop_size=crop_size,
            num_examples=num_examples,
            index_file="index_generator/provided/mastr_files.txt")
