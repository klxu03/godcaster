import os

import json

import math

import av

import numpy as np

from torch.utils.data import Dataset


class GodCasterDatasetProcess(Dataset):

    def __init__(self, video_folder: str, caption_folder: str, sample_frames: int = 400) -> None:
        super().__init__()

        self.video_folder = video_folder

        self.caption_folder = caption_folder

        self.sample_frames = sample_frames

        self.video_files = sorted([os.path.join(self.video_folder, x) for x in os.listdir(self.video_folder)])

        self.caption_files = sorted([os.path.join(self.caption_folder, x) for x in os.listdir(self.caption_folder)])

    def __len__(self):
        return len(self.video_files)
    
    def __getitem__(self, index):
        video_file_to_be_processed = self.video_files[index]

        with open(self.caption_files[index]) as f:
            captions = f.read()

        container = av.open(video_file_to_be_processed)

        number_of_frames = container.streams.video[0].frames

        indicies = self.sample_frame_indices(self.sample_frames, number_of_frames)

        return self.read_video_av(container, indicies), captions
    
    def sample_frame_indices(self, sample_frames, frame_length_of_clip):
        initial_burst_sample = list(range(50))

        step_size = (2/3)*(frame_length_of_clip - 50)/(sample_frames - 50)

        num_frames_third = math.floor((1/3)*(frame_length_of_clip - 50)/step_size)

        num_frames_two_third = math.ceil((2/3)*(frame_length_of_clip - 50)/(2*step_size))

        third_partition_entities = np.floor(np.linspace(50, 50+(frame_length_of_clip - 50)//3, num=num_frames_third))

        two_third_partition_entities = np.floor(np.linspace(50+(frame_length_of_clip - 50)//3, frame_length_of_clip, num=num_frames_two_third))

        return initial_burst_sample + list(third_partition_entities) + list(two_third_partition_entities)
       

    def read_video_pyav(self, container, indices):
        '''
        Decode the video with PyAV decoder.
        Args:
            container (`av.container.input.InputContainer`): PyAV container.
            indices (`List[int]`): List of frame indices to decode.
        Returns:
            result (np.ndarray): np array of decoded frames of shape (num_frames, height, width, 3).
        '''
        frames = []
        container.seek(0)
        start_index = indices[0]
        end_index = indices[-1]
        for i, frame in enumerate(container.decode(video=0)):
            if i > end_index:
                break
            if i >= start_index and i in indices:
                frames.append(frame)
        return np.stack([x.to_ndarray(format="rgb24") for x in frames]) 


    
    