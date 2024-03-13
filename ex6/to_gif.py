#simple script to convert a mp4 file to a gif

import os
import imageio
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--conf', type=int, default=1, help='Configuration for losses (1-3)')

args = parser.parse_args()
conf = args.conf
conf = 1

assert conf in {1,2,3}

video_path = fr'./results/soapbox/{conf}/slow_motion_video.mp4'
output_path = fr'./results/soapbox/{conf}/slow_motion_video.gif'

# read the video file
video = imageio.get_reader(video_path)

#get number of farmes in video
n_frames = len(list(video.iter_data()))
print(n_frames)

# write the gif file
imageio.mimsave(output_path, [frame for frame in video.iter_data()], duration=20)
print(f'Gif saved at {output_path}')

#also save the original video as a gif

video_path = fr'./results/soapbox/{conf}/original_video.mp4'
output_path = fr'./results/soapbox/{conf}/original_video.gif'

# read the video file
video = imageio.get_reader(video_path)
n_frames = len(list(video.iter_data()))

print(n_frames)
# write the gif file
imageio.mimsave(output_path, [frame for frame in video.iter_data()], duration=10)

print(f'Original video gif saved at {output_path}')
