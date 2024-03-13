#simple script to convert mp4 to folder with pngs

import os
import imageio
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--conf', type=int, default=1, help='Configuration for losses (1-3)')
args = parser.parse_args()
conf = args.conf
conf = 2

assert conf in {1,2,3}

video_path = fr'./results/soapbox/{conf}/slow_motion_video.mp4'
output_path = fr'./results/soapbox/{conf}/slow_motion_video'


# read the video file
video = imageio.get_reader(video_path)

# get number of farmes in video

n_frames = len(list(video.iter_data()))
print(n_frames)

#save folder with pngs
os.makedirs(output_path, exist_ok=True)
for i, frame in enumerate(video.iter_data()):
    imageio.imsave(os.path.join(output_path, f'im-{i}.png'), frame)
print(f'Pngs saved at {output_path}')
#also save the original video as a folder with pngs

video_path = fr'./results/soapbox/{conf}/original_video.mp4'
output_path = fr'./results/soapbox/{conf}/original_video'

# read the video file
video = imageio.get_reader(video_path)
n_frames = len(list(video.iter_data()))

print(n_frames)

# save folder with pngs
os.makedirs(output_path, exist_ok=True)
for i, frame in enumerate(video.iter_data()):
    imageio.imsave(os.path.join(output_path, f'im-{i}.png'), frame)
print(f'Original video pngs saved at {output_path}')

