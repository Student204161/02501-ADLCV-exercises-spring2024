#simple script that compares absolute difference between two folders with pngs

import os
import imageio
import argparse
import numpy as np
parser = argparse.ArgumentParser()

parser.add_argument('--conf1', type=int, default=1, help='Configuration for losses (1-3)')
parser.add_argument('--conf2', type=int, default=1, help='Configuration for losses (1-3)')
args = parser.parse_args()
conf1 = args.conf1
conf2 = args.conf2
conf1 = 2
conf2 = 3

assert conf1 in {1,2,3}
assert conf2 in {1,2,3}
assert conf1 != conf2

pngs_path1 = fr'./results/soapbox/{conf1}/predicted_frames'
pngs_path2 = fr'./results/soapbox/{conf2}/predicted_frames'

pngs1 = os.listdir(pngs_path1)
pngs2 = os.listdir(pngs_path2)

assert len(pngs1) == len(pngs2)

for i in range(len(pngs1)):
    im1 = imageio.imread(os.path.join(pngs_path1, pngs1[i]))
    im2 = imageio.imread(os.path.join(pngs_path2, pngs2[i]))
    assert im1.shape == im2.shape
    assert im1.dtype == im2.dtype
    #convert to float32

    im1 = im1.astype('float32')
    im2 = im2.astype('float32')

    abs_img = np.abs(im1 - im2)

    #to uint8 again
    abs_img = abs_img.astype('uint8')

    #save path should be in a new folder under soapbox
    save_path = fr'./results/soapbox/abs_diff_{conf1}_{conf2}'
    os.makedirs(save_path, exist_ok=True)
    #absolute difference between the two images shown as an image
    imageio.imsave(os.path.join(save_path, f'abs_diff_{i}.png'), abs_img)
print(f'Absolute difference between the two folders saved at {save_path}')