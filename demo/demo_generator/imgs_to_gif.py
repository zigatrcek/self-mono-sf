import imageio
import os

images = []

vis_dir = "results/mods/vis"
for i in range(0, 90):
    images.append(imageio.imread(os.path.join(vis_dir, f'vis_{i:02d}.png')))
print(images[0].shape)
imageio.mimsave(os.path.join(vis_dir, 'test.gif'), images)
