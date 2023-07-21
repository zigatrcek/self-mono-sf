from pathlib import Path
def iter_subtree(path, index_path, dataset='modd2'):
    """Find all frames in MODD2 dataset and index them.
    """

    # make Path object from input string
    path = Path(path)
    with open(index_path, 'w+') as index_file:
        # iter the directory
        for p in path.iterdir():
            if p.is_dir():
                # enter `frames` directory
                for d in p.iterdir():
                    if d.name in ['framesRectified', 'images', 'frames']:
                        # iter the frames directory
                        for i, f in enumerate(sorted(d.iterdir())):
                            if dataset == 'lars':
                                index_file.write(f'{p.name} {f.name.split(".")[0]}\n')
                            elif dataset == 'modd2':
                                if i % 2 == 0:
                                # write to file
                                    index_file.write(f'{p.name} {f.name.split(".")[0][:-1]}\n')
                            elif dataset == 'mods':
                                if i % 2 == 0:
                                # write to file
                                    index_file.write(f'{p.name} {f.name.split(".")[0][:-1]}\n')


            else:
                raise FileNotFoundError()


def iter_mastr(path, index_path):
    """Find all frames in the MAStR dataset and index them.

    Args:
        path (str): Path to the MAStR dataset.
        index_path (str): Where the index file should be written.
    """


    # make Path object from input string
    path = Path(path)
    with open(index_path, 'w+') as index_file:
        # iter the directory
        for f in sorted(path.iterdir()):
            index_file.write(f'{f.name}\n')

if __name__ == '__main__':
    # iter_subtree('../../../data/modd2/rectified_video_data/', './provided/modd2_files.txt', dataset='modd2')
    # iter_subtree('../../../data/mods/sequences/', './provided/mods_files.txt', dataset='mods')
    iter_subtree('/storage/datasets/modb_raw/sequences', './provided/modb_raw_files.txt', dataset='mods')

    # iter_subtree('../../../data/LaRS_v0.9.3/', './provided/lars_files.txt', dataset='lars')
    # iter_mastr('../../../data/mastr1325/MaSTr1325_images_512x384/', './provided/mastr_files.txt')
