import numpy as np
import cv2
from matplotlib import pyplot as plt

if __name__ == '__main__':
    disp_mods_path = '/home/ziga/self-mono-sf/eval/monosf_modd2_selfsup_mods_test/flow/kope101/00004150L_flow.png'
    disp_kitti_path = '/home/ziga/self-mono-sf/eval/monosf_kitti_selfsup_mods_test/flow/kope101/00004150L_flow.png'
    cre_path = '/storage/private/student-vicos/mods_cre/sequences/kope101-00004130-00004650/frames/00004150L.png'
    img_path = '/storage/private/student-vicos/mods_rectified/sequences/kope101-00004130-00004650/frames/00004150L.jpg'

    fig, axs = plt.subplots(4, 4)
    axs = axs.flatten()
    ax = axs[0]
    ax.axis('off')
    ax.set_title('Originalna slika')
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    ax.imshow(img)

    ax = axs[1]
    ax.axis('off')
    ax.set_title('CRE')
    cre = cv2.imread(cre_path, cv2.IMREAD_UNCHANGED).astype(np.float32) / 32
    ax.imshow(cre, cmap='plasma')

    ax = axs[2]
    ax.axis('off')
    ax.set_title('Kitti')
    img = cv2.imread(disp_kitti_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    ax.imshow(img, cmap='plasma')

    ax = axs[3]
    ax.axis('off')
    ax.set_title('MODD2')
    img = cv2.imread(disp_mods_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    ax.imshow(img, cmap='plasma')

    disp_mods_path = '/home/ziga/self-mono-sf/eval/monosf_modd2_selfsup_mods_test/flow/kope102/00005810L_flow.png'
    disp_kitti_path = '/home/ziga/self-mono-sf/eval/monosf_kitti_selfsup_mods_test/flow/kope102/00005810L_flow.png'
    cre_path = '/storage/private/student-vicos/mods_cre/sequences/kope102-00005520-00006840/frames/00005810L.png'
    img_path = '/storage/private/student-vicos/mods_rectified/sequences/kope102-00005520-00006840/frames/00005810L.jpg'

    ax = axs[4]
    ax.axis('off')
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    ax.imshow(img)

    ax = axs[5]
    ax.axis('off')
    cre = cv2.imread(cre_path, cv2.IMREAD_UNCHANGED).astype(np.float32) / 32
    ax.imshow(cre, cmap='plasma')

    ax = axs[6]
    ax.axis('off')
    img = cv2.imread(disp_kitti_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    ax.imshow(img, cmap='plasma')

    ax = axs[7]
    ax.axis('off')
    img = cv2.imread(disp_mods_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    ax.imshow(img, cmap='plasma')

    disp_mods_path = '/home/ziga/self-mono-sf/eval/monosf_modd2_selfsup_mods_test/flow/kope103/00007950L_flow.png'
    disp_kitti_path = '/home/ziga/self-mono-sf/eval/monosf_kitti_selfsup_mods_test/flow/kope103/00007950L_flow.png'
    cre_path = '/storage/private/student-vicos/mods_cre/sequences/kope103-00007600-00008500/frames/00007950L.png'
    img_path = '/storage/private/student-vicos/mods_rectified/sequences/kope103-00007600-00008500/frames/00007950L.jpg'

    ax = axs[8]
    ax.axis('off')
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    ax.imshow(img)

    ax = axs[9]
    ax.axis('off')
    cre = cv2.imread(cre_path, cv2.IMREAD_UNCHANGED).astype(np.float32) / 32
    ax.imshow(cre, cmap='plasma')

    ax = axs[10]
    ax.axis('off')
    img = cv2.imread(disp_kitti_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    ax.imshow(img, cmap='plasma')

    ax = axs[11]
    ax.axis('off')
    img = cv2.imread(disp_mods_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    ax.imshow(img, cmap='plasma')

    disp_mods_path = '/home/ziga/self-mono-sf/eval/monosf_modd2_selfsup_mods_test/flow/kope103/00007670L_flow.png'
    disp_kitti_path = '/home/ziga/self-mono-sf/eval/monosf_kitti_selfsup_mods_test/flow/kope103/00007670L_flow.png'
    cre_path = '/storage/private/student-vicos/mods_cre/sequences/kope103-00007600-00008500/frames/00007670L.png'
    img_path = '/storage/private/student-vicos/mods_rectified/sequences/kope103-00007600-00008500/frames/00007670L.jpg'

    ax = axs[12]
    ax.axis('off')
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    ax.imshow(img)

    ax = axs[13]
    ax.axis('off')
    cre = cv2.imread(cre_path, cv2.IMREAD_UNCHANGED).astype(np.float32) / 32
    ax.imshow(cre, cmap='plasma')

    ax = axs[14]
    ax.axis('off')
    img = cv2.imread(disp_kitti_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    ax.imshow(img, cmap='plasma')

    ax = axs[15]
    ax.axis('off')
    img = cv2.imread(disp_mods_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    ax.imshow(img, cmap='plasma')





    plt.tight_layout()
    plt.show()


# if __name__ == '__main__':
#     photo_folder = '2011_09_26/2011_09_26_drive_0020_sync'
#     input_photo_name = '0000000052'
#     photo_name = '000000'
#     dir_title_list = [
#         ('kitti/full_model_kitti_ft', 'Fine-tuned KITTI'),
#         ('kitti/modd2_fulldata_15_epochs_checkpoint_latest', 'MODD2 - 15 epochs'),
#     ]
#     l = len(dir_title_list)
#     f, axarr = plt.subplots(l+1, 2, figsize=(6, 5))
#     axarr = axarr.flatten()
#     input_path = f'../data/kitti_data/{photo_folder}/image_02/data/{input_photo_name}.jpg'
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
