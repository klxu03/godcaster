from . import WhisperX
import json
import os

class WhisperXRunner:
    def __init__(self, video_paths, output_dir, model_name="large-v3", compute_type="float16", batch_size=16):
        self.video_paths = video_paths
        self.output_dir = output_dir
        self.whisper = WhisperX(model_name, compute_type, batch_size)
    
    def run(self):
        for video_path in self.video_paths:
            filename = os.path.splitext(os.path.basename(video_path))[0]
            text = self.whisper.video_transcribe(video_path)
            with open(os.path.join(self.output_dir,f'{filename}.json'), "w") as f:
                json.dump(text, f, indent=2)