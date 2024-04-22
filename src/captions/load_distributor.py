import os
from moviepy.editor import VideoFileClip
import math

def load_distribute(max_index, ind):
    dir_path = "/scratch/kxu39/merged/"
    subdirs = [dir for dir in os.listdir(dir_path) if os.path.isdir(os.path.join(dir_path, dir))]
    # print("subdirs", subdirs)
    num_vids = len(subdirs)
    # print("num_vids", num_vids)

    vid_len = {}
    for subdir in subdirs:
        vid_list = [vid for vid in os.listdir(os.path.join(dir_path, subdir)) if vid.endswith('.mp4')]
        # print("vid_list", vid_list)
        sum = 0
        full_subdir_path = dir_path + subdir
        for i in range(len(vid_list)):
            with VideoFileClip(f"{full_subdir_path}/{vid_list[i]}") as temp_vid:
                dur = temp_vid.duration
                sum += dur
        vid_len[full_subdir_path] = sum

    vid_with_len = []
    sum = 0
    for k, v in vid_len.items():
        vid_with_len.append((k, v))
        sum += v

    threshold = math.ceil(sum / float(max_index + 1))
    # print("threshold", threshold, "sum", sum)

    vid_with_len.sort(key=lambda x: x[1], reverse=True)
    # print("vid_with_len", vid_with_len)
    
    indexes = [[]]
    curr_sum = 0

    for i in range(num_vids):
        dur = vid_with_len[0][1]
        if curr_sum + dur > threshold:
            print("curr_sum", curr_sum, "dur", dur, "threshold", threshold)
            min_ind = min((index for index, (word, number) in enumerate(vid_with_len) if number >= threshold - curr_sum), key=lambda idx: vid_with_len[idx][1], default=None)
            curr_sum = 0
            print("min_ind", min_ind)
            indexes[-1].append(vid_with_len[min_ind][0])
            vid_with_len.pop(min_ind)

            if len(indexes) < max_index:
                indexes.append([])
        else:
            curr_sum += dur
            indexes[-1].append(vid_with_len[0][0])
            vid_with_len.pop(0)

        print("i", i, "indexes", indexes)

    while len(indexes) <= max_index:
        indexes.append([])

    # print("indexes", indexes)
    print(f"indexes[ind] for ind {ind} is {indexes[ind]}")
    return indexes[ind]

if __name__ == "__main__":
    load_distribute(2, 0)