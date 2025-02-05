from __future__ import absolute_import, division, print_function

import os.path
import torch
import torch.utils.data as data
import numpy as np

from torchvision import transforms as vision_transforms
from .common import read_image_as_byte, read_modd2_calib_into_dict, read_annotation
from .common import kitti_crop_image_list, kitti_adjust_intrinsic
import logging


class MODD2_Raw(data.Dataset):
    def __init__(self,
                 args,
                 images_root=None,
                 flip_augmentations=True,
                 preprocessing_crop=True,
                 crop_size=[950, 1224],
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



        filename_list = [line.rstrip().split(' ') for line in index_file.readlines()]
        logging.debug(f'Number of images: {len(filename_list)}')
        self._image_list = []
        ext = '.jpg'

        for item in filename_list:
            scene = item[0]
            # take two consecutive frames
            idx_src = item[1]
            idx_tgt = '%.8d' % (int(idx_src) + 1)
            name_l1 = os.path.join(images_root, scene, 'framesRectified', idx_src) + 'L' + ext
            name_l2 = os.path.join(images_root, scene, 'framesRectified', idx_tgt) + 'L' + ext
            name_r1 = os.path.join(images_root, scene, 'framesRectified', idx_src) + 'R' + ext
            name_r2 = os.path.join(images_root, scene, 'framesRectified', idx_tgt) + 'R' + ext

            if all([
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


        if num_examples > 0:
            # TODO shuffling doesn't do anything since the data is only loaded in the first epoch.
            # logging.info(f'Using {num_examples} examples, shuffling...')
            # np.random.shuffle(self._image_list)
            self._image_list = self._image_list[:num_examples]

        self._size = len(self._image_list)

        ## loading calibration matrix
        self.intrinsic_dict_l = {}
        self.intrinsic_dict_r = {}
        self.intrinsic_dict_l, self.intrinsic_dict_r = read_modd2_calib_into_dict(path_dir)

        self._to_tensor = vision_transforms.Compose([
            vision_transforms.ToPILImage(),
            vision_transforms.transforms.ToTensor()
        ])

    def __getitem__(self, index):
        index = index % self._size

        # read images and flow
        # im_l1, im_l2, im_r1, im_r2
        img_list_np = [read_image_as_byte(img) for img in self._image_list[index]]
        # logging.info(f'annotations_list: {annotations_list}')
        # for annotation in annotations_list[0]['annotations']:
        #     logging.info(f'annotation: {annotation}')

        # example filename
        im_l1_filename = self._image_list[index][0]
        # name of sequence directory
        sequence = os.path.basename(os.path.dirname(os.path.dirname(im_l1_filename)))
        k_l1 = torch.from_numpy(self.intrinsic_dict_l[sequence]).float()
        k_r1 = torch.from_numpy(self.intrinsic_dict_r[sequence]).float()

        # input size
        h_orig, w_orig, _ = img_list_np[0].shape
        input_im_size = torch.from_numpy(np.array([h_orig, w_orig])).float()

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
            k_l1, k_r1 = kitti_adjust_intrinsic(k_l1, k_r1, crop_info)

        # to tensors
        img_list_tensor = [self._to_tensor(img) for img in img_list_np]

        im_l1 = img_list_tensor[0]
        im_l2 = img_list_tensor[1]
        im_r1 = img_list_tensor[2]
        im_r2 = img_list_tensor[3]



        common_dict = {
            "index": index,
            "basename": f'{sequence}-{index}',
            "datename": sequence,
            "input_size": input_im_size
        }

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
            }
            example_dict.update(common_dict)

        else:
            example_dict = {
                "input_l1": im_l1,
                "input_r1": im_r1,
                "input_l2": im_l2,
                "input_r2": im_r2,
                "input_k_l1": k_l1,
                "input_k_r1": k_r1,
                "input_k_l2": k_l1,
                "input_k_r2": k_r1,
            }
            example_dict.update(common_dict)

        return example_dict

    def __len__(self):
        return self._size



class MODD2_Train(MODD2_Raw):
    def __init__(self,
                 args,
                 root,
                 flip_augmentations=True,
                 preprocessing_crop=True,
                 crop_size=[950, 1224],
                 num_examples=-1):
        super(MODD2_Train, self).__init__(
            args,
            images_root=root,
            flip_augmentations=flip_augmentations,
            preprocessing_crop=preprocessing_crop,
            crop_size=crop_size,
            num_examples=num_examples,
            index_file="index_generator/generated/modd2_train.txt")


class MODD2_Valid(MODD2_Raw):
    def __init__(self,
                 args,
                 root,
                 flip_augmentations=False,
                 preprocessing_crop=False,
                 crop_size=[950, 1224],
                 num_examples=-1):
        super(MODD2_Valid, self).__init__(
            args,
            images_root=root,
            flip_augmentations=flip_augmentations,
            preprocessing_crop=preprocessing_crop,
            crop_size=crop_size,
            num_examples=num_examples,
            index_file="index_generator/generated/modd2_valid.txt")

class MODD2_Visualisation(MODD2_Raw):
    def __init__(self,
                 args,
                 root,
                 flip_augmentations=False,
                 preprocessing_crop=False,
                 crop_size=[950, 1224],
                 num_examples=-1):
        super(MODD2_Visualisation, self).__init__(
            args,
            images_root=root,
            flip_augmentations=flip_augmentations,
            preprocessing_crop=preprocessing_crop,
            crop_size=crop_size,
            num_examples=num_examples,
            index_file="index_generator/provided/modd2_visualisation.txt")




class Inference_Raw(data.Dataset):
    def __init__(self,
                 args,
                 images_root=None,
                 flip_augmentations=True,
                 preprocessing_crop=True,
                 crop_size=[950, 1224],
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

        filename_list = [line.rstrip().split(' ') for line in index_file.readlines()]
        logging.info(f'Number of images: {len(filename_list)}')
        self._image_list = []
        ext = '.jpg'

        for item in filename_list:
            subfolder = item[0]
            # take two consecutive frames
            filename_without_idx = '_'.join(item[1].split('_')[:-1])
            idx_tgt = item[1].split('_')[-1]
            idx_src = f'%.{len(idx_tgt)}d' % (int(idx_tgt) - 1)
            filename_src = filename_without_idx + '_' + idx_src + ext
            filename_tgt = filename_without_idx + '_' + idx_tgt + ext
            logging.debug(f'filename_src: {filename_src}')
            name_l1 = os.path.join(images_root, subfolder, 'images_seq', filename_src)
            name_l2 = os.path.join(images_root, subfolder, 'images_seq', filename_tgt)
            logging.debug(f'name_l1: {name_l1}')
            logging.debug(f'name_l2: {name_l2}')

            # logging.info(f'{name_l1} exists: {os.path.isfile(name_l1)}\n{name_l2} exists: {os.path.isfile(name_l2)}')
            if os.path.isfile(name_l1) and os.path.isfile(name_l2):
                self._image_list.append([name_l1, name_l2])
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

        # read images and flow
        # im_l1, im_l2, im_r1, im_r2
        img_list_np = [read_image_as_byte(img) for img in self._image_list[index]]

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
            k_l1, k_r1 = kitti_adjust_intrinsic(k_l1, k_r1, crop_info)

        # to tensors
        img_list_tensor = [self._to_tensor(img) for img in img_list_np]

        im_l1 = img_list_tensor[0]
        im_l2 = img_list_tensor[1]
        # im_r1 = img_list_tensor[2]
        # im_r2 = img_list_tensor[3]

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
            # should never get here in inference mode
            _, _, ww = im_l1.size()
            im_l1_flip = torch.flip(im_l1, dims=[2])
            im_l2_flip = torch.flip(im_l2, dims=[2])
            # im_r1_flip = torch.flip(im_r1, dims=[2])
            # im_r2_flip = torch.flip(im_r2, dims=[2])

            k_l1[0, 2] = ww - k_l1[0, 2]
            k_r1[0, 2] = ww - k_r1[0, 2]

            example_dict = {
                "input_l1": im_l1_flip,
                # "input_r1": im_l1_flip,
                "input_l2": im_l2_flip,
                # "input_r2": im_l2_flip,
                "input_k_l1": k_l1,
                # "input_k_r1": k_l1,
                "input_k_l2": k_l1,
                # "input_k_r2": k_l1,
            }
            example_dict.update(common_dict)

        else:
            example_dict = {
                "input_l1": im_l1,
                # "input_r1": im_r1,
                "input_l2": im_l2,
                # "input_r2": im_r2,
                "input_k_l1": k_l1,
                "input_k_r1": k_r1,
                "input_k_l2": k_l1,
                "input_k_r2": k_r1,
            }
            example_dict.update(common_dict)

        return example_dict

    def __len__(self):
        return self._size



class MODD2_Inference(Inference_Raw):
    def __init__(self,
                 args,
                 root,
                 flip_augmentations=False,
                 preprocessing_crop=False,
                 crop_size=[950, 1224],
                 num_examples=-1):
        super(MODD2_Inference, self).__init__(
            args,
            images_root=root,
            flip_augmentations=flip_augmentations,
            preprocessing_crop=preprocessing_crop,
            crop_size=crop_size,
            num_examples=num_examples,
            index_file="index_generator/provided/lars_files.txt")
