from WhisperX import WhisperX
import json
import os
import sys

class WhisperXRunner:
    def __init__(self, model_name="large-v3", compute_type="float16", batch_size=16):
        self.model_name = model_name
        self.compute_type = compute_type
        self.batch_size = batch_size
    
    def run(self, video_dir, output_dir):
        whisper = WhisperX(self.model_name, self.compute_type, self.batch_size)

        for video_file in os.listdir(video_dir):
            name_ext = os.path.splitext(os.path.basename(video_file))
            if name_ext[1] == '.mp4':
                video_path = os.path.join(video_dir, video_file)
                print('Processing', video_path)
                text = whisper.video_transcribe(video_path)
                with open(os.path.join(output_dir,f'{name_ext[0]}.json'), "w") as f:
                    json.dump(text, f, indent=2)

if __name__ == "__main__":
    dir = sys.argv[1]
    runner = WhisperXRunner(model_name="large-v3", compute_type="float16", batch_size=16)
    runner.run(f"{dir}/", f"{dir}/")