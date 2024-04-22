import WhisperX
import json
import os

class WhisperXRunner:
    def __init__(self, video_dir, output_dir, model_name="large-v3", compute_type="float16", batch_size=16):
        self.video_dir = video_dir
        self.output_dir = output_dir
        self.whisper = WhisperX(model_name, compute_type, batch_size)
    
    def run(self):
        for video_file in os.listdir(self.video_dir):
            name_ext = os.path.splitext(os.path.basename(video_file))
            if name_ext[1] == '.mp4':
                video_path = os.path.join(self.video_dir, video_file)
                print('Processing', video_path)
                text = self.whisper.video_transcribe(video_path)
                with open(os.path.join(self.output_dir,f'{name_ext[0]}.json'), "w") as f:
                    json.dump(text, f, indent=2)