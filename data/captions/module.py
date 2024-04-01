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

    def download_yt_vid(self, yt_url, filepath):
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

    def yt_transcribe(self, yt_url: str, task="transcribe"):
        with tempfile.TemporaryDirectory() as tmpdirname:
            filepath = os.path.join(tmpdirname, "video.mp4")
            self.download_yt_vid(yt_url, filepath)
            with open(filepath, "rb") as f:
                inputs = f.read()

        inputs = ffmpeg_read(inputs, self.pipe.feature_extractor.sampling_rate)
        inputs = {"array": inputs, "sampling_rate": self.pipe.feature_extractor.sampling_rate}

        res = self.pipe(inputs, batch_size=self.BATCH_SIZE, generate_kwargs={"task": task}, return_timestamps=True)
        print(f"Captions:\n{res['text']}")
        return res

import whisperx
import gc
import subprocess

class WhisperX:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.BATCH_SIZE = 16
        compute_type = "float16"

        self.model = whisperx.load_model("large-v3", self.device, compute_type=compute_type)
        self.whisper = Whisper()

    def convert_mp4_to_mp3(self, video_filepath: str, audio_filepath: str, ar=44100, ac=2, b_a="192k"):
        if not os.path.exists(video_filepath):
            raise FileNotFoundError(f"File not found: {video_filepath}")

        command = f"ffmpeg -i {video_filepath} -vn -ar {ar} -ac {ac} -b:a {b_a} {audio_filepath}"
        try:
            subprocess.check_output(command, stderr=subprocess.STDOUT, shell=True)
        except subprocess.CalledProcessError as e:
            print(f"An error occurred while converting {video_filepath} to {audio_filepath}")
            print(f"Error message: {e.output.decode()}")

    def yt_transcribe(self, yt_url: str):
        with tempfile.TemporaryDirectory() as tmpdirname:
            video_filepath = os.path.join(tmpdirname, "video.mp4")
            audio_filepath = os.path.join(tmpdirname, "audio.mp4")
            self.whisper.download_yt_vid(yt_url, video_filepath)
            self.convert_mp4_to_mp3(video_filepath, audio_filepath)

            audio = whisperx.load_audio(audio_filepath)
            result = self.model.transcribe(audio, batch_size=self.BATCH_SIZE)
            print(f"Before alignment:\n{result['segments']}")

            gc.collect(); torch.cuda.empty_cache(); del self.model; # free up memory

        model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=self.device)
        result = whisperx.align(result["segments"], model_a, metadata, audio, self.device, return_char_alignments=False)

        print(result["segments"]) # after alignment

        gc.collect(); torch.cuda.empty_cache(); del model_a; # free up memory

        return result["segments"]

        """
        Further speaker diarization can be done with the following code:

        https://github.com/m-bain/whisperX
        # 3. Assign speaker labels
        diarize_model = whisperx.DiarizationPipeline(use_auth_token=YOUR_HF_TOKEN, device=device)

        # add min/max number of speakers if known
        diarize_segments = diarize_model(audio)
        # diarize_model(audio, min_speakers=min_speakers, max_speakers=max_speakers)

        result = whisperx.assign_word_speakers(diarize_segments, result)
        print(diarize_segments)
        print(result["segments"]) # segments are now assigned speaker IDs
        """