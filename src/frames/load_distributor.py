import os
from moviepy.editor import VideoFileClip

def load_distribute(max_index, ind):
    dir_path = "/scratch/kxu39/merged/"
    vid_list = os.listdir(dir_path)

    vid_with_len = []

    sum = 0
    for i in range(len(vid_list)):
        if vid_list[i] == ".gitkeep":
            continue

        vid_list[i] = dir_path + vid_list[i]
        # print("vid_list", vid_list[i])
        with VideoFileClip(vid_list[i]) as temp_vid:
            dur = temp_vid.duration
            sum += dur

        vid_with_len.append((vid_list[i], dur))
    threshold = sum / (max_index + 1)

    vid_with_len.sort(key=lambda x: x[1], reverse=True)
    
    indexes = [[]]
    curr_ind = 0
    curr_sum = 0

    for i, (vid, dur) in enumerate(vid_with_len):
        indexes[curr_ind].append(vid)
        curr_sum += dur
        if curr_sum > threshold:
            curr_sum = 0
            curr_ind += 1
            indexes.append([])

    while len(indexes) <= max_index:
        indexes.append([])

    # print("indexes", indexes)

    return indexes[ind]