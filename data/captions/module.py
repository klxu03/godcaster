import torch

import yt_dlp as youtube_dl
from transformers import pipeline
from transformers.pipelines.audio_utils import ffmpeg_read

import tempfile
import os

class Whisper:
    def __init__(self):
        device = 0 if torch.cuda.is_available() else "cpu"
        MODEL_NAME = "openai/whisper-large-v3"

        self.pipe = pipeline(
            task="automatic-speech-recognition",
            model=MODEL_NAME,
            chunk_length_s=30,
            device=device
        )

        self.YT_LENGTH_LIMIT_H = 2 # limit of 2 hour YouTube files
        self.BATCH_SIZE = 7

    def download_yt_audio(self, yt_url, filepath):
        info_loader = youtube_dl.YoutubeDL()

        try:
            info = info_loader.extract_info(yt_url, download=False)
        except youtube_dl.utils.DownloadError:
            raise RuntimeError("Error: YT DL is unable to extract info from the video")

        file_length = info["duration_string"]
        file_h_m_s = file_length.split(":")
        file_h_m_s = [int(sub_length) for sub_length in file_h_m_s]
        
        # if the video is less than 3 parts, add 0s to the beginning populating mins, then hours
        while len(file_h_m_s) < 3:
            file_h_m_s.insert(0, 0)
        
        if file_h_m_s[0] > self.YT_LENGTH_LIMIT_H:
            raise IOError(f"Maximum YouTube length is {self.YT_LENGTH_LIMIT_H}, got {file_h_m_s[0]} YouTube video.")
        
        ydl_opts = {"outtmpl": filepath, "format": "worstvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best"}
        
        with youtube_dl.YoutubeDL(ydl_opts) as ydl:
            try:
                ydl.download([yt_url])
            except youtube_dl.utils.ExtractorError as err:
                raise RuntimeError("Error: YT DL is unable to download video")

    def yt_transcribe(self, yt_url: str, task: str) -> str:
        with tempfile.TemporaryDirectory() as tmpdirname:
            filepath = os.path.join(tmpdirname, "video.mp4")
            self.download_yt_audio(yt_url, filepath)
            with open(filepath, "rb") as f:
                inputs = f.read()

        inputs = ffmpeg_read(inputs, self.pipe.feature_extractor.sampling_rate)
        inputs = {"array": inputs, "sampling_rate": self.pipe.feature_extractor.sampling_rate}

        text = self.pipe(inputs, batch_size=self.BATCH_SIZE, generate_kwargs={"task": task}, return_timestamps=True)["text"]

        return text