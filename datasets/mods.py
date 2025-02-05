from __future__ import absolute_import, division, print_function

import os.path
import torch
import torch.utils.data as data
import numpy as np

from torchvision import transforms as vision_transforms
from .common import read_image_as_byte, read_mods_calib_into_dict, read_annotation, get_disp
from .common import kitti_crop_image_list, kitti_adjust_intrinsic
import logging


class Mods_Base(data.Dataset):
    def __init__(self,
                 args,
                 images_root=None,
                 flip_augmentations=True,
                 preprocessing_crop=True,
                 crop_size=[928, 1248],
                 num_examples=-1,
                 index_file=None):

        self._args = args
        self._seq_len = 1
        self._flip_augmentations = flip_augmentations
        self._preprocessing_crop = preprocessing_crop
        self._crop_size = crop_size
        self.seg = True

        path_dir = os.path.dirname(os.path.realpath(__file__))
        path_index_file = os.path.join(path_dir, index_file)
        # seg_dir = ('/home/bogosort/diploma/data/mods_wasr')
        seg_dir = ('/storage/private/student-vicos/mods_wasr')

        disp_dir = ('/storage/private/student-vicos/mods_cre/sequences')

        # calib_dir = ('../data/calibration')
        calib_dir = ('/storage/private/student-vicos/mods/calibration')

        # log index file
        logging.info(f'Index file: {path_index_file}')
        logging.info(f'Images root: {images_root}')

        if not os.path.exists(path_index_file):
            raise ValueError("Index File '%s' not found!", path_index_file)
        index_file = open(path_index_file, 'r')

        # loading image -----------------------------------
        if not os.path.isdir(images_root):
            raise ValueError(f"Image directory '{images_root}' not found!")

        if self.seg and not os.path.isdir(seg_dir):
            raise ValueError(f"Segmentation directory '{seg_dir}' not found!")

        filename_list = [line.rstrip().split(' ')
                         for line in index_file.readlines()]
        # shuffle list
        np.random.shuffle(filename_list)

        logging.info(f'Number of images: {len(filename_list)}')
        self._image_list = []
        self._ignore_masks = []
        self._gt_list = []

        for item in filename_list:
            ext = '.jpg'
            scene = item[0]
            # take two consecutive frames
            idx_src = item[1]
            idx_tgt = '%.8d' % (int(idx_src) + 10)
            # print(f'idx_src: {idx_src}, idx_tgt: {idx_tgt}')
            # print(os.path.join(images_root, scene, 'frames', idx_src) + 'L' + ext)
            name_l1 = os.path.join(
                images_root, scene, 'frames', idx_src) + 'L' + ext
            name_l2 = os.path.join(
                images_root, scene, 'frames', idx_tgt) + 'L' + ext
            name_r1 = os.path.join(
                images_root, scene, 'frames', idx_src) + 'R' + ext
            name_r2 = os.path.join(
                images_root, scene, 'frames', idx_tgt) + 'R' + ext
            name_ignore_mask = os.path.join(
                images_root, scene, 'ignore_mask.png')


            if self.seg:
                ext = '.png'
                seg_l1 = os.path.join(
                    seg_dir, scene, idx_src) + 'L' + ext
                disp = os.path.join(
                    disp_dir, scene, 'frames', idx_tgt) + 'L.png'


            if not self.seg and all([
                os.path.isfile(name_l1),
                os.path.isfile(name_l2),
                os.path.isfile(name_r1),
                os.path.isfile(name_r2),
            ]):
                # logging.info(f'All files exist.')
                self._image_list.append([
                    name_l1,
                    name_l2,
                    name_r1,
                    name_r2,
                ])
            elif self.seg and all([
                os.path.isfile(name_l1),
                os.path.isfile(name_l2),
                os.path.isfile(name_r1),
                os.path.isfile(name_r2),
                os.path.isfile(seg_l1),
                os.path.isfile(disp),
            ]):
                self._image_list.append([
                    name_l1,
                    name_l2,
                    name_r1,
                    name_r2,
                    seg_l1,
                ])
                self._gt_list.append(disp)
            if os.path.isfile(name_ignore_mask):
                self._ignore_masks.append(name_ignore_mask)
            else:
                self._ignore_masks.append(None)

        if num_examples > 0:
            self._image_list = self._image_list[:num_examples]

        self._size = len(self._image_list)
        logging.info(f'Number of examples: {self._size}')

        # loading calibration matrix
        self.intrinsic_dict_l = {}
        self.intrinsic_dict_r = {}
        self.intrinsic_dict_t = {}
        self.intrinsic_dict_l, self.intrinsic_dict_r, self.intrinsic_dict_t = read_mods_calib_into_dict(
            calib_dir)

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
        img_list_np = [read_image_as_byte(img)
                       for img in self._image_list[index]]
        img_list_np.append(get_disp(self._gt_list[index]))
        # logging.info(f'img_list_np[0].shape: {img_list_np[0].shape}')
        if self._ignore_masks[index] is not None:
            ignore_mask = read_image_as_byte(self._ignore_masks[index])[:, :, :3]
            if self.seg:
                seg_l1 = img_list_np[4]
                disp = img_list_np[5]
            masked = []
            for img in img_list_np[:4]:
                img_masked = img * np.logical_not(ignore_mask)
                masked.append(img_masked)

            img_list_np = masked
            if self.seg:
                img_list_np.append(seg_l1)
                img_list_np.append(disp)
            img_list_np.append(ignore_mask)



        # example filename
        im_l1_filename = self._image_list[index][0]
        im_l1_displayname = self._image_list[index][0].split(
            '/')[-1].split('.')[0]
        im_l2_filename = self._image_list[index][1].split(
            '/')[-1].split('.')[0]
        logging.debug(f'im_l2_filename: {im_l2_filename}')
        # name of sequence directory
        sequence = os.path.basename(os.path.dirname(
            os.path.dirname(im_l1_filename))).split('-')[0]
        # logging.info(f'Sequence: {sequence}')

        k_l1 = torch.from_numpy(self.intrinsic_dict_l[sequence]).float()
        k_r1 = torch.from_numpy(self.intrinsic_dict_r[sequence]).float()
        translation_vector = torch.from_numpy(self.intrinsic_dict_t[sequence]).float()

        # input size
        h_orig, w_orig, _ = img_list_np[0].shape
        input_im_size = torch.from_numpy(np.array([h_orig, w_orig])).float()

        # read annotations
        # ann_l1, ann_l2

        # cropping
        if self._preprocessing_crop:

            # get starting positions
            crop_height = self._crop_size[0]
            crop_width = self._crop_size[1]
            x = np.random.uniform(0, w_orig - crop_width + 1)
            y = np.random.uniform(0, h_orig - crop_height + 1)
            crop_info = [int(x), int(y), int(
                x + crop_width), int(y + crop_height)]

            # cropping images and adjust intrinsic accordingly
            img_list_np = kitti_crop_image_list(img_list_np, crop_info)
            k_l1, k_r1 = kitti_adjust_intrinsic(k_l1, k_r1, crop_info)

        # to tensors
        img_list_tensor = [self._to_tensor(img) for img in img_list_np]

        im_l1 = img_list_tensor[0]
        im_l2 = img_list_tensor[1]
        im_r1 = img_list_tensor[2]
        im_r2 = img_list_tensor[3]

        common_dict = {
            'sequence': sequence,
            'img_name': im_l1_displayname,  # take first image as name
            "index": index,
            "basename": f'{sequence}-{index}',
            "datename": sequence,
            "input_size": input_im_size,
            "img_fullname": im_l1_filename,
        }
        logging.debug(f'common_dict: {common_dict}')

        # random flip
        if self._flip_augmentations is True and torch.rand(1) > 0.5:
            _, _, ww = im_l1.size()
            im_l1_flip = torch.flip(im_l1, dims=[2])
            im_l2_flip = torch.flip(im_l2, dims=[2])
            im_r1_flip = torch.flip(im_r1, dims=[2])
            im_r2_flip = torch.flip(im_r2, dims=[2])

            k_l1[0, 2] = ww - k_l1[0, 2]
            k_r1[0, 2] = ww - k_r1[0, 2]

            example_dict = {
                "input_l1": im_r1_flip,
                "input_r1": im_l1_flip,
                "input_l2": im_r2_flip,
                "input_r2": im_l2_flip,
                "input_k_l1": k_r1,
                "input_k_r1": k_l1,
                "input_k_l2": k_r1,
                "input_k_r2": k_l1,
                "input_t": translation_vector,
            }
            example_dict.update(common_dict)
        else:
            example_dict = {
                "input_l1": im_l1,
                "input_l2": im_l2,
                "input_r1": im_r1,
                "input_r2": im_r2,
                "input_k_l1": k_l1,
                "input_k_r1": k_r1,
                "input_k_l2": k_l1,
                "input_k_r2": k_r1,
                "input_t": translation_vector,
            }
            if self.seg:
                example_dict["seg_l1"] = img_list_tensor[4]
                example_dict["disp"] = img_list_tensor[5]

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


class Mods_Full(Mods_Base):
    def __init__(self,
                 args,
                 root,
                 flip_augmentations=False,
                 preprocessing_crop=True,
                 crop_size=[928, 1248],
                 num_examples=-1):
        super(Mods_Full, self).__init__(
            args,
            images_root=root,
            flip_augmentations=flip_augmentations,
            preprocessing_crop=preprocessing_crop,
            crop_size=crop_size,
            num_examples=num_examples,
            index_file="index_generator/provided/mods_files.txt")


class Mods_Train(Mods_Base):
    def __init__(self,
                 args,
                 root,
                 flip_augmentations=True,
                 preprocessing_crop=True,
                 crop_size=[950, 1224],
                 num_examples=-1):
        super(Mods_Train, self).__init__(
            args,
            images_root=root,
            flip_augmentations=flip_augmentations,
            preprocessing_crop=preprocessing_crop,
            crop_size=crop_size,
            num_examples=num_examples,
            index_file="index_generator/generated/mods_train.txt")


class Mods_Valid(Mods_Base):
    def __init__(self,
                 args,
                 root,
                 flip_augmentations=False,
                 preprocessing_crop=False,
                 crop_size=[950, 1224],
                 num_examples=-1):
        super(Mods_Valid, self).__init__(
            args,
            images_root=root,
            flip_augmentations=flip_augmentations,
            preprocessing_crop=preprocessing_crop,
            crop_size=crop_size,
            num_examples=num_examples,
            index_file="index_generator/generated/mods_val.txt")


class Mods_Test(Mods_Base):
    def __init__(self,
                 args,
                 root,
                 flip_augmentations=False,
                 preprocessing_crop=False,
                 crop_size=[950, 1224],
                 num_examples=-1):
        super(Mods_Test, self).__init__(
            args,
            images_root=root,
            flip_augmentations=flip_augmentations,
            preprocessing_crop=preprocessing_crop,
            crop_size=crop_size,
            num_examples=num_examples,
            index_file="index_generator/provided/mods_files.txt")
