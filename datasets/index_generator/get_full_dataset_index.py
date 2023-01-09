from pathlib import Path
def iter_subtree(path, index_path):
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
                    if d.name == 'frames':
                        # iter the frames directory
                        for i, f in enumerate(sorted(d.iterdir())):
                            if i % 2 == 0:
                            # write to file
                                index_file.write(f'{p.name} {f.name.split(".")[0][:-1]}\n')



            else:
                raise FileNotFoundError()

if __name__ == '__main__':
    iter_subtree('../../modd2/video_data/', './provided/modd2_files.txt')
