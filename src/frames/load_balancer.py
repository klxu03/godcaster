import os
from moviepy.editor import VideoFileClip

def load_balance(total, ind):
    vid_list = os.listdir("/scratch/kxu39")
    print(vid_list)

    vid_with_len = []

    sum = 0
    for i in range(vid_list):
        vid_list[i] = "/scratch/kxu39/" + vid_list[i]
        dur = VideoFileClip(vid_list[i]).duration
        sum += dur
        vid_with_len.append((vid_list[i], dur))
    threshold = sum / total

    vid_with_len.sort(key=lambda x: x[1], reverse=True)
    
    indexes = [None] * total
    curr_ind = 0
    curr_sum = 0

    for i, (vid, dur) in enumerate(vid_with_len):
        indexes[curr_ind].append(vid)
        if curr_sum + dur > threshold:
            curr_sum = 0
            curr_ind += 1

    return indexes[ind]