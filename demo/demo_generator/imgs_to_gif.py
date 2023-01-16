import imageio
import os

images = []

vis_dir = "results/modd2/vis"
for i in range(0, 39):
    images.append(imageio.imread(os.path.join(vis_dir, f'000561_{i:02d}.png')))
imageio.imsave(os.path.join(vis_dir, '00561.gif'), images)
