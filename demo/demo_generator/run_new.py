import os

import skimage.io as io
from skimage.color import rgb2gray
# from skimage.color import lab2rgb

import open3d as o3d
import numpy as np
import torch
import math
from matplotlib import pyplot as plt

from utils_misc import flow_to_png_middlebury, read_png_flow, read_png_disp
from utils_misc import numpy2torch, pixel2pts_ms

width_to_focal = dict()
width_to_focal[1278] = 9.7210644631744879e+02
# width_to_focal[1242] = 721.5377
# width_to_focal[1241] = 718.856
# width_to_focal[1238] = 718.3351
# width_to_focal[1226] = 707.0912
# width_to_focal[1224] = 707.0493

cam_center_dict = dict()
cam_center_dict[1278] = [639, 749]
# cam_center_dict[1242] = [6.095593e+02, 1.728540e+02]
# cam_center_dict[1241] = [6.071928e+02, 1.852157e+02]
# cam_center_dict[1238] = [6.003891e+02, 1.815122e+02]
# cam_center_dict[1226] = [6.018873e+02, 1.831104e+02]
# cam_center_dict[1224] = [6.040814e+02, 1.805066e+02]


########
sampling = [4,20,25,35,40]
imgflag = 1 # 0 is image, 1 is flow
########



def get_pcd(img_idx, image_dir, result_dir, tt):

    idx_curr = '%08d' % (img_idx)

    im1_np0 = (io.imread(os.path.join(image_dir, idx_curr + "L.jpg")) / np.float32(255.0))[300:, :, :]

    flo_f_np0 = read_png_flow(os.path.join(result_dir, "flow", 'kope103', idx_curr + "L_10.png"))[300:, :, :]
    disp1_np0 = read_png_disp(os.path.join(result_dir, "disp_0", 'kope103', idx_curr + "L_10.png"))[300:, :, :]
    disp2_np0 = read_png_disp(os.path.join(result_dir, "disp_1", 'kope103', idx_curr + "L_10.png"))[300:, :, :]
    # plt.imshow(disp2_np0)
    # plt.show()
    # exit()

    im1 = numpy2torch(im1_np0).unsqueeze(0)
    disp1 = numpy2torch(disp1_np0).unsqueeze(0)
    print(disp1.shape)
    disp_diff = numpy2torch(disp2_np0).unsqueeze(0)
    flo_f = numpy2torch(flo_f_np0).unsqueeze(0)

    _, _, hh, ww = im1.size()

    ## Intrinsic
    focal_length = width_to_focal[ww]
    cx = cam_center_dict[ww][0]
    cy = cam_center_dict[ww][1]

    k1_np = np.array([[focal_length, 0, cx], [0, focal_length, cy], [0, 0, 1]])
    k1 = numpy2torch(k1_np)

    # Forward warping Pts1 using disp_change and flow
    pts1 = pixel2pts_ms(disp1, k1)
    pts1_warp = pixel2pts_ms(disp_diff, k1, flo_f)
    sf = pts1_warp - pts1


    ## Composing Image
    im1_np0_g = np.repeat(np.expand_dims(rgb2gray(im1_np0), axis=2), 3, axis=2)
    flow = torch.cat((sf[:, 0:1, :, :], sf[:, 2:3, :, :]), dim=1).data.cpu().numpy()[0, :, :, :]
    flow_img = flow_to_png_middlebury(flow) / np.float32(255.0)
    print(pts1.shape)
    print(flow_img.shape)


    if imgflag == 0:
        flow_img = im1_np0
    else:
        flow_img = (flow_img * 0.75 + im1_np0_g * 0.25)

    ## Crop
    max_crop = (60, 0.7, 82)
    min_crop = (-60, -20, 0)

    x1 = -60
    x2 = 60
    y1 = 0.7
    y2 = -20
    z1 = 80
    z2 = 0
    pp1 = np.array([[x1, y1, z1]])
    pp2 = np.array([[x1, y1, z2]])
    pp3 = np.array([[x1, y2, z1]])
    pp4 = np.array([[x1, y2, z2]])
    pp5 = np.array([[x2, y1, z1]])
    pp6 = np.array([[x2, y1, z2]])
    pp7 = np.array([[x2, y2, z1]])
    pp8 = np.array([[x2, y2, z2]])
    bb_pts = np.concatenate((pp1, pp2, pp3, pp4, pp5, pp6, pp7, pp8), axis=0)
    wp = np.array([[1.0, 1.0, 1.0]])
    bb_colors = np.concatenate((wp, wp, wp, wp, wp, wp, wp, wp), axis=0)

    ## Open3D Vis
    pts1_tform = pts1 + sf*tt
    pts1_np = np.transpose(pts1_tform[0].view(3, -1).data.numpy(), (1, 0))
    pts1_np = np.concatenate((pts1_np, bb_pts), axis=0)
    pts1_color = np.reshape(flow_img, (hh * ww, 3))
    print(f'shape: {pts1_color.shape}, mean: {pts1_color.mean()}, min: {pts1_color.min()}, max: {pts1_color.max()}')
    print(f'shape: {pts1_np.shape}, mean: {pts1_np.mean()}, min: {pts1_np.min()}, max: {pts1_np.max()}')
    pts1_color = np.concatenate((pts1_color, bb_colors), axis=0)

    pcd1 = o3d.geometry.PointCloud()
    pcd1.points = o3d.utility.Vector3dVector(pts1_np)
    pcd1.colors = o3d.utility.Vector3dVector(pts1_color)

    # pcd1 = pcd1.crop(bbox)

    return pcd1

i = 0

def custom_vis(imglist, kitti_data_dir, result_dir, vis_dir):
    vis = o3d.visualization.Visualizer()
    vis.create_window()

    # ctr = vis.get_view_control()
    # ctr.set_zoom(0.11999999999999984)
    # ctr.set_front([ -0.027364235563851778, 0.27175389648221898, -0.96197766001021956 ])
    # ctr.set_lookat([ 0.40990976922313455, -6.0531471549110982, 6.7731456531852841 ])
    # ctr.set_up([ 0, -0.96236368380624848, -0.27157101181444943 ])

    def next_frame(_):
        global i
        vis.clear_geometries()
        init_pcd = get_pcd(imglist[i], kitti_data_dir, result_dir, 0)
        vis.add_geometry(init_pcd)
        ctr = vis.get_view_control()
        ctr.set_zoom(0.11999999999999984)
        ctr.set_front([ -0.027364235563851778, 0.27175389648221898, -0.96197766001021956 ])
        ctr.set_lookat([ 0.40990976922313455, -6.0531471549110982, 6.7731456531852841 ])
        ctr.set_up([ 0, -0.96236368380624848, -0.27157101181444943 ])
        img = vis.capture_screen_float_buffer(True)
        img = np.asarray(img) * 255
        img = img.astype(np.uint8)
        io.imsave(os.path.join(vis_dir, f'vis_{i:02d}.png'), img, check_contrast=False)
        i = i + 1

    vis.register_animation_callback(next_frame)
    vis.run()
    next_frame()
    vis.destroy_window()




########################################################################

# kitti_data_dir = "demo/demo_generator/kitti_img"    ## raw KITTI image
# result_dir = "demo/demo_generator/results"          ## disp_0, disp_1, flow
# vis_dir = "demo/demo_generator/vis"                 ## visualization output folder

kitti_data_dir = "/home/bogosort/diploma/self-mono-sf/demo/demo_generator/mods/kope103-00007600-00008500/frames"    ## raw KITTI image
result_dir = "/home/bogosort/diploma/self-mono-sf/demo/demo_generator/mods/uncropped_preds"          ## disp_0, disp_1, flow
vis_dir = "results/mods/vis"                 ## visualization output folder
if not os.path.exists(vis_dir):
    os.makedirs(vis_dir)

imglist = []

for ii in range(7600, 8600, 10):
    imglist.append(ii)


custom_vis(imglist, kitti_data_dir, result_dir, vis_dir)
