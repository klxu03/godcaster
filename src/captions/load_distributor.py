import os
from moviepy.editor import VideoFileClip
import math

def load_distribute(max_index, ind):
    dir_path = "/scratch/kxu39/merged/"
    subdirs = [dir for dir in os.listdir(dir_path) if os.path.isdir(os.path.join(dir_path, dir))]
    print("subdirs", subdirs)
    num_vids = len(subdirs)

    vid_len = {}
    for subdir in subdirs:
        vid_list = [vid for vid in os.listdir(dir_path + subdir)]
        sum = 0
        for i in range(len(vid_list)):
            with VideoFileClip(vid_list[i]) as temp_vid:
                dur = temp_vid.duration
                sum += dur
        vid_len[subdir] = sum

    vid_with_len = []
    sum = 0
    for k, v in vid_len.items():
        vid_with_len.append((k, v))
        sum += v

    threshold = math.ceil(sum / float(max_index + 1))

    vid_with_len.sort(key=lambda x: x[1], reverse=True)
    
    indexes = [[]]
    curr_sum = 0

    for i in range(num_vids):
        dur = vid_with_len[0][1]
        if curr_sum + dur > threshold:
            curr_sum = 0
            min_ind = min((index for index, (word, number) in enumerate(vid_list) if number >= threshold - curr_sum), key=lambda idx: vid_list[idx][1], default=None)
            indexes[-1].append(vid_list[min_ind][0])
            vid_list.pop(min_ind)
            indexes.append([])
        else:
            curr_sum += dur
            indexes[-1].append(vid_list[0][0])
            vid_list.pop(0)

    while len(indexes) <= max_index:
        indexes.append([])

    return indexes[ind]