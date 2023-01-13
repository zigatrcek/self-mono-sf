import numpy as np
import cv2
from matplotlib import pyplot as plt

# if __name__ == '__main__':
#     photo_folder = 'kope81-00-00000560-00001080'
#     input_photo_name = '00000561L'
#     photo_name = 'kope81-00-00000560-00001080-0'
#     dir_title_list = [
#         ('monosf_kitti_ft', 'Fine-tuned KITTI'),
#         ('full_model_eigen', 'Eigen split'),
#         ('modd2_fulldata_15_epochs_checkpoint_latest', 'MODD2 - 15 epochs'),
#     ]
#     l = len(dir_title_list)
#     f, axarr = plt.subplots(l+1, 2, figsize=(6, 5))
#     axarr = axarr.flatten()
#     input_path = f'../data/modd2/rectified_video_data/{photo_folder}/framesRectified/{input_photo_name}.jpg'
#     input_image = cv2.imread(input_path)
#     input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
#     axarr[0].set_title('Input image')
#     axarr[0].imshow(input_image)
#     axarr[0].axis('off')
#     axarr[1].set_visible(False)

#     # load images from dir list
#     for i, (dir, title) in enumerate(dir_title_list):
#         disp_path = f'../eval/visualisation/{dir}/disp_0/{photo_name}_disp.jpg'
#         disp = cv2.imread(disp_path)
#         disp = cv2.cvtColor(disp, cv2.COLOR_BGR2RGB)

#         axarr[2*i + 2].set_title(f'{title} - disp')
#         axarr[2*i + 2].imshow(disp)
#         axarr[2*i + 2].axis('off')


#         flow_path = f'../eval/visualisation/{dir}/flow/{photo_name}_flow.png'
#         flow = cv2.imread(flow_path)
#         flow = cv2.cvtColor(flow, cv2.COLOR_BGR2RGB)

#         axarr[2*i + 3].set_title(f'{title} - flow')
#         axarr[2*i + 3].imshow(flow)
#         axarr[2*i + 3].axis('off')
#     plt.tight_layout()
#     plt.savefig(f'../eval/comparisons/compare{input_photo_name}.png', dpi=300)


if __name__ == '__main__':
    photo_folder = '2011_09_26/2011_09_26_drive_0020_sync'
    input_photo_name = '0000000052'
    photo_name = '000000'
    dir_title_list = [
        ('kitti/full_model_kitti_ft', 'Fine-tuned KITTI'),
        ('kitti/modd2_fulldata_15_epochs_checkpoint_latest', 'MODD2 - 15 epochs'),
    ]
    l = len(dir_title_list)
    f, axarr = plt.subplots(l+1, 2, figsize=(6, 5))
    axarr = axarr.flatten()
    input_path = f'../data/kitti_data/{photo_folder}/image_02/data/{input_photo_name}.jpg'
    input_image = cv2.imread(input_path)
    input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
    axarr[0].set_title('Input image')
    axarr[0].imshow(input_image)
    axarr[0].axis('off')
    axarr[1].set_visible(False)

    # load images from dir list
    for i, (dir, title) in enumerate(dir_title_list):
        disp_path = f'../eval/visualisation/{dir}/disp_0/{photo_name}_disp.jpg'
        disp = cv2.imread(disp_path)
        disp = cv2.cvtColor(disp, cv2.COLOR_BGR2RGB)

        axarr[2*i + 2].set_title(f'{title} - disp')
        axarr[2*i + 2].imshow(disp)
        axarr[2*i + 2].axis('off')


        flow_path = f'../eval/visualisation/{dir}/flow/{photo_name}_flow.png'
        flow = cv2.imread(flow_path)
        flow = cv2.cvtColor(flow, cv2.COLOR_BGR2RGB)

        axarr[2*i + 3].set_title(f'{title} - flow')
        axarr[2*i + 3].imshow(flow)
        axarr[2*i + 3].axis('off')
    plt.tight_layout()
    plt.savefig(f'../eval/comparisons/compare{input_photo_name}.png', dpi=300)
