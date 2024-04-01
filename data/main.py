from captions.module import Whisper, WhisperX

YT_URL = "https://www.youtube.com/watch?v=r4qS0WyH2Lk" 

import json

def main():
    # populate audio
    whisper = WhisperX()
    text = whisper.yt_transcribe(YT_URL)

    with open("captions/transcript.json", "w") as f:
        json.dump(text, f, indent=2)

if __name__ == "__main__":
    main()