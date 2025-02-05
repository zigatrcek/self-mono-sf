from __future__ import absolute_import, division, print_function

import numpy as np
from glob import glob
import os
import copy
from collections import defaultdict

dir_path = os.path.dirname(os.path.realpath(__file__))


def extract_kitti_benchmark_scene():
    eigen_test_file = os.path.join(dir_path, 'provided/train_mapping.txt')
    if not os.path.exists(eigen_test_file):
        raise ValueError("KITTI Train File '%s' not found!", eigen_test_file)
    eigen_test = open(eigen_test_file, 'r')

    scene_name_set = set()
    filename_list = [line.split(' ') for line in eigen_test.readlines()]
    for item in filename_list:
        if len(item) != 3:
            continue
        _, scene_name, _ = item
        scene_name_set.add(scene_name)

    scene_name_set = sorted(scene_name_set)

    with open(os.path.join(dir_path, 'generated/kitti_test_scenes.txt'), 'w') as tf:
        for item in scene_name_set:
            tf.write('%s\n' % item)


def extract_eigen_test_scene():
    eigen_test_file = os.path.join(dir_path, 'provided/eigen_test_files.txt')
    if not os.path.exists(eigen_test_file):
        raise ValueError("Eigen Test File '%s' not found!", eigen_test_file)
    eigen_test = open(eigen_test_file, 'r')

    scene_name_set = set()
    filename_list = [line.split(' ') for line in eigen_test.readlines()]
    for item in filename_list:
        _, scene_name, _, _, _ = item[0].split('/')
        scene_name_set.add(scene_name)

    scene_name_set = sorted(scene_name_set)

    with open(os.path.join(dir_path, 'generated/eigen_test_scenes.txt'), 'w') as tf:
        for item in scene_name_set:
            tf.write('%s\n' % item)


def extract_eigen_test_kitti_benchmark_scene():

    # Eigen
    eigen_test_file = os.path.join(dir_path, 'txt/eigen_test_files.txt')
    if not os.path.exists(eigen_test_file):
        raise ValueError("Eigen Test File '%s' not found!", eigen_test_file)
    eigen_test = open(eigen_test_file, 'r')

    scene_name_set = set()
    filename_list = [line.split(' ') for line in eigen_test.readlines()]
    for item in filename_list:
        _, scene_name, _, _, _ = item[0].split('/')
        scene_name_set.add(scene_name)

    # KITTI
    kitti_test_file = os.path.join(dir_path, 'txt/train_mapping.txt')
    if not os.path.exists(kitti_test_file):
        raise ValueError("KITTI Train File '%s' not found!", kitti_test_file)
    kitti_test = open(kitti_test_file, 'r')

    filename_list = [line.split(' ') for line in kitti_test.readlines()]
    for item in filename_list:
        if len(item) != 3:
            continue
        _, scene_name, _ = item
        scene_name_set.add(scene_name)

    scene_name_set = sorted(scene_name_set)

    with open(os.path.join(dir_path, 'txt/eigen_kitti_test_scenes.txt'), 'w') as tf:
        for item in scene_name_set:
            tf.write('%s\n' % item)


class CollectDataList(object):
    def __init__(self, dataset_dir, split='eigen', sequence_length=1):

        # filename / variable definition
        excluded_frames_file = dir_path + '/provided/excluded_frames.txt'
        self.date_list = ['2011_09_26', '2011_09_28',
                          '2011_09_29', '2011_09_30', '2011_10_03']
        self.sequence_length = sequence_length
        self.dataset_dir = dataset_dir
        self.excluded_frames = []
        self.train_frames = []
        self.num_train = -1
        self.split = split

        # parsing data
        test_scene_file = dir_path + '/generated/' + self.split + '_test_scenes.txt'
        with open(test_scene_file, 'r') as f:
            test_scenes = f.readlines()
        self.test_scenes = [t.rstrip() for t in test_scenes]

        # self.collect_excluded_frames(excluded_frames_file)
        self.collect_train_frames()

    # Exclude KITTI flow frames
    def collect_excluded_frames(self, excluded_frames_file):
        with open(excluded_frames_file, 'r') as f:
            frames = f.readlines()

        for fr in frames:
            date, drive, frame_id = fr.rstrip().split(' ')
            self.excluded_frames.append(drive + ' ' + frame_id)

    def collect_train_frames(self):
        all_frames = []
        for date in self.date_list:
            drive_set = os.listdir(self.dataset_dir + date + '/')
            for dr in drive_set:
                drive_dir = os.path.join(self.dataset_dir, date, dr)

                if os.path.isdir(drive_dir):
                    if dr in self.test_scenes:
                        continue
                    img_dir = os.path.join(drive_dir, 'image_02', 'data')
                    num_images = len(glob(img_dir + '/*.jpg'))
                    for n in range(num_images-self.sequence_length):
                        frame_id = '%.10d' % n
                        all_frames.append(dr + ' ' + frame_id)

        for s in self.excluded_frames:
            try:
                all_frames.remove(s)
            except:
                pass

        self.train_frames = all_frames
        self.num_train = len(self.train_frames)
        print(self.num_train)
        with open(os.path.join(dir_path, 'generated', self.split + '_full.txt'), 'w') as tf:
            for item in self.train_frames:
                tf.write('%s\n' % item)


class SplitTrainVal(object):
    def __init__(self, dataset_dir, file_name, seq_len, alias):

        # Variable definition
        self.dataset_dir = dataset_dir
        self.seq_len = seq_len
        self.file_name = file_name
        self.alias = alias
        self.train_set = set()
        self.ref_val_set = set()
        self.eff_ref_val_set = set()

        # build dataset
        self.build_data()
        self.write_dataset_file()

    def build_data(self):

        with open(os.path.join(dir_path, self.file_name), 'r') as tf:
            all_frames = tf.readlines()

        all_frames = [ff.rstrip() for ff in all_frames]
        np.random.seed(8964)
        np.random.shuffle(all_frames)

        # 1. pick ref. val. image
        n_val_images = int(0.03 * (len(all_frames)))
        self.ref_val_set = set(all_frames[:n_val_images])
        self.eff_ref_val_set = copy.deepcopy(self.ref_val_set)

        # 2. create a set ('val. images' + '+seq_len val. images' + '-seq_len val. images')
        for fr in self.ref_val_set:
            drive, frame_id = fr.rstrip().split(' ')
            ref_id = int(frame_id)

            for ii in range(ref_id - self.seq_len, ref_id + self.seq_len + 1):
                if ii < 0:
                    continue
                fr_alias = drive + ' ' + '%.10d' % ii
                if fr_alias not in all_frames:
                    continue

                self.eff_ref_val_set.add(drive + ' ' + '%.10d' % ii)

        # 3. refine the train set (exclude frames that are included in the eff_ref_val_set)
        self.eff_ref_train_set = set(all_frames) - self.eff_ref_val_set
        self.train_set = copy.deepcopy(self.eff_ref_train_set)
        for fr in self.eff_ref_train_set:
            drive, frame_id = fr.rstrip().split(' ')
            ref_id = int(frame_id)

            for ii in range(ref_id - self.seq_len, ref_id + self.seq_len + 1):
                if ii < 0:
                    self.train_set.remove(fr)
                    break
                fr_alias = drive + ' ' + '%.10d' % ii
                if fr_alias in self.eff_ref_val_set:
                    self.train_set.remove(fr)
                    break

        # 4. training ref. image = Total dataset - the set above

        print('all frame ', len(all_frames))
        print('eff ref train set', len(self.eff_ref_train_set))
        print('train set ', len(self.train_set))
        print('eff ref val set', len(self.eff_ref_val_set))
        print('ref val set', len(self.ref_val_set))

    def write_dataset_file(self):
        with open(os.path.join(dir_path, self.alias + '_our_train.txt'), 'w') as tf:
            for item in sorted(self.train_set):
                tf.write('%s\n' % item)

        with open(os.path.join(dir_path, self.alias + '_our_valid.txt'), 'w') as vf:
            for item in sorted(self.ref_val_set):
                vf.write('%s\n' % item)


class SplitTrainVal_discard_last_frame(object):
    def __init__(self, dataset_dir, file_name, seq_len, alias):

        # Variable definition
        self.dataset_dir = dataset_dir
        self.seq_len = seq_len
        self.file_name = file_name
        self.alias = alias
        self.train_set = set()
        self.ref_val_set = set()
        self.eff_ref_val_set = set()

        # build dataset
        self.build_data()
        self.write_dataset_file()

    def build_data(self):

        with open(os.path.join(dir_path, self.file_name), 'r') as tf:
            all_frames = tf.readlines()

        all_frames = [ff.rstrip() for ff in all_frames]
        np.random.seed(8964)
        np.random.shuffle(all_frames)

        self.train_set = copy.deepcopy(all_frames)

        # 1. Discard last frame
        for fr in all_frames:
            drive, frame_id = fr.rstrip().split(' ')
            ref_id = int(frame_id)

            fr_alias = drive + ' ' + '%.10d' % (ref_id + 1)
            if fr_alias not in all_frames:
                self.train_set.remove(fr)

        # 4. training ref. image = Total dataset - the set above

        print('all frame ', len(all_frames))
        print('train set ', len(self.train_set))

    def write_dataset_file(self):
        with open(os.path.join(dir_path, self.alias + '_our_train.txt'), 'w') as tf:
            for item in sorted(self.train_set):
                tf.write('%s\n' % item)


class SplitTrainVal_even(object):
    def __init__(self, dataset_dir, file_name, chunk_size, alias):
        self.dataset_dir = dataset_dir
        self.chunk_size = chunk_size
        self.file_name = file_name
        self.alias = alias
        self.splits = {
            'train': 0.5,
            'val': 0.125,
            'test': 0.375
        }

        self.split_data()
        self.write_index_files()

    def split_data(self):
        """Split data into train, validation, and test sets.
        Makes sure that there are no overlapping frames between the splits,
        since the model takes two consecutive frames as input.
        We assume that the `torch.Dataset` takes the frame and the next according to index.
        """
        # read all frames
        with open(os.path.join(dir_path, self.file_name), 'r') as tf:
            lines = [line.rstrip().split(' ') for line in tf.readlines()]

        sequences = defaultdict(list)
        [sequences[line[0]].append(int(line[1])) for line in lines]

        for seq in sequences.values():
            assert self.check_consecutive(seq, step=1)

        # split sequences into chunks
        chunks = []
        for key, seq in sequences.items():
            chunks += list(self.chunks_without_every_nth_element(seq,
                           self.chunk_size, key))


        # split chunks into train, validation, and test sets
        np.random.seed(8964)
        np.random.shuffle(chunks)

        train_size = int(self.splits['train'] * len(chunks))
        val_size = int(self.splits['val'] * len(chunks))

        self.sets = {
            'train': chunks[:train_size],
            'val': chunks[train_size:train_size + val_size],
            'test': chunks[train_size + val_size:]
        }


    @staticmethod
    def check_consecutive(l, step=1):
        return l == list(range(l[0], l[-1] + step, step))

    @staticmethod
    def chunks_without_every_nth_element(lst, n, key):
        """
        Yield successive n-sized chunks from lst.
        Drop last element of each chunk to avoid overlapping frames.
        """
        for i in range(0, len(lst), n):
            yield (key, lst[i:i + n - 1])


    def write_index_files(self):
        for split, frames in sorted(self.sets.items(), key=lambda x: x[0]):
            with open(os.path.join(dir_path, 'generated', self.alias + '_' + split + '.txt'), 'w+') as f:
                for frame in sorted(frames, key=lambda x: x[0]):
                    seq_name = frame[0]
                    for frame_id in frame[1]:
                        f.write('%s %s\n' % (seq_name, '%.8d' % frame_id))

def main():

    chunk_size = 10


    dataset_dir = 'mods/sequences/'
    SplitTrainVal_even(dataset_dir=dataset_dir, file_name='provided/modb_raw_files.txt',
                       chunk_size=chunk_size, alias='modb_raw')


main()
