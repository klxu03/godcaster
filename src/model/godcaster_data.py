import os

import json

import math

from typing import List

import cv2

import numpy as np

from torch.utils.data import Dataset

class GodCasterDataset(Dataset):

    def __init__(self, video_folder: str, caption_folder: str) -> None:
        super().__init__()

        self.video_folder = video_folder

        self.caption_folder = caption_folder

        self.video_files = sorted(self.read_folder(self.video_folder, ".mp4"))

        self.caption_files = sorted(self.read_folder(self.caption_folder, ".json"))

        index_lengths = []

        for caption_file in self.caption_files:
            with open(caption_file, "r") as f:
                data = json.load(f)
            index_lengths.append(len(data))
        
        index_cumulative_lengths = list(np.cumsum(index_lengths))

        self.len = sum(index_cumulative_lengths)
    
        self.index_map = dict(zip(index_cumulative_lengths, list(zip(self.video_files, self.caption_files))))
        
    
    def read_folder(self, folder: str, ext: str) -> List[str]:
        total_files = [] 

        for root, dirs, files in os.walk(folder):
            for name in files:
                if ext in name:
                    total_files.append(os.path.join(root, name))
        
        return total_files

    def __len__(self):
        return self.len
    
    def __getitem__(self, index):

        og_index = index

        while index not in self.index_map:
            index -= 1

        file_pair = self.index_map[index]

        with open(file_pair[1]) as f:
            captions = f.read()

        container = cv2.VideoCapture(file_pair[0])

        FPS = container.get(cv2.CAP_PROP_FPS)

        sentence_index = og_index - index

        sentence_start_frame = math.ceil(captions[sentence_index]["start"] * FPS)

        if sentence_start_frame < 230:
            num_copies = 230//sentence_start_frame

            left_over = 230 % sentence_start_frame

            indicies = [0]*(num_copies + left_over)

            for i in range(1, sentence_start_frame):
                indicies += [i]*num_copies

        elif sentence_start_frame < FPS * 60:
            indicies = list(np.linspace(0, sentence_start_frame, num=230))
        else:
            indicies = self.sample_frame_indices(sentence_start_frame, FPS)

        frames = []

        for index in indicies:
            container.set(cv2.CAP_PROP_POS_FRAMES, index)

            _, frame = container.read()

            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))


        return frames, " ".join([x["text"] for x in captions[:sentence_index]]), captions[sentence_index]["text"]
    
    # Assumes we are over a minute long length
    def sample_frame_indices(self, frame_length_of_clip, FPS):
        initial_rate = FPS // 5

        ret = [i*initial_rate for i in range(50)]

        new_tot = frame_length_of_clip - math.ceil(10 * FPS)
        
        first_third = np.ceil(np.linspace(0, new_tot // 3, num=90) + math.ceil(10 * FPS))

        second_third = np.ceil(np.linspace(new_tot//3, new_tot, num=90) + math.ceil(10 * FPS))

        return ret + list(first_third) + list(second_third)
       


    
    