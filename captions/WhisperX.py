import torch
import tempfile
import os
import whisperx
import subprocess

class WhisperX:
    def __init__(self, model_name="large-v3", compute_type="float16", batch_size = 16):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.BATCH_SIZE = batch_size
        self.model = whisperx.load_model(model_name, self.device, compute_type=compute_type)

    def convert_mp4_to_mp3(self, video_filepath: str, audio_filepath: str, ar=44100, ac=2, b_a="192k"):
        if not os.path.exists(video_filepath):
            raise FileNotFoundError(f"File not found: {video_filepath}")

        command = f"ffmpeg -i {video_filepath} -vn -ar {ar} -ac {ac} -b:a {b_a} {audio_filepath}"
        try:
            subprocess.check_output(command, stderr=subprocess.STDOUT, shell=True)
        except subprocess.CalledProcessError as e:
            print(f"An error occurred while converting {video_filepath} to {audio_filepath}")
            print(f"Error message: {e.output.decode()}")

    def video_transcribe(self, video_filepath: str):
        with tempfile.TemporaryDirectory() as tmpdirname:
            audio_filepath = os.path.join(tmpdirname, "audio.mp4")
            self.convert_mp4_to_mp3(video_filepath, audio_filepath)

            audio = whisperx.load_audio(audio_filepath)
            result = self.model.transcribe(audio, batch_size=self.BATCH_SIZE)

        model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=self.device)
        result = whisperx.align(result["segments"], model_a, metadata, audio, self.device, return_char_alignments=False)

        return result["segments"]
