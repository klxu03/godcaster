import torch
import tempfile
import os
import whisperx
import subprocess
import ffmpeg

class WhisperX:
    def __init__(self, model_name="large-v3", compute_type="float16", batch_size = 16, hf_token=None):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.BATCH_SIZE = batch_size
        self.model = whisperx.load_model(model_name, self.device, compute_type=compute_type)
        self.model_align, self.metadata = whisperx.load_align_model(language_code="en", device=self.device)
        self.hf_token = hf_token

    def convert_mp4_to_mp3(self, video_filepath: str, audio_filepath: str, ar=44100, ac=2, b_a="192k"):
        if not os.path.exists(video_filepath):
            raise FileNotFoundError(f"File not found: {video_filepath}")
        ffmpeg.input(video_filepath).audio.output(audio_filepath, **{'b:a': b_a,'ar':ar, 'ac':ac}).run(quiet=True)

    def video_transcribe(self, video_filepath: str):
        with tempfile.TemporaryDirectory() as tmpdirname:
            audio_filepath = os.path.join(tmpdirname, "audio.mp4")
            self.convert_mp4_to_mp3(video_filepath, audio_filepath)

            audio = whisperx.load_audio(audio_filepath)
            result = self.model.transcribe(audio, batch_size=self.BATCH_SIZE)
        result = whisperx.align(result["segments"], self.model_align, self.metadata, audio, self.device, return_char_alignments=False)

        diarize_model = whisperx.DiarizationPipeline(use_auth_token=self.hf_token, device="cuda")
        diarize_segments = diarize_model(audio)

        result = whisperx.assign_word_speakers(diarize_segments, result)
        
        return result["segments"]
