from captions.module import Whisper

YT_URL = "https://www.youtube.com/watch?v=r4qS0WyH2Lk" 

import json

def main():
    # populate audio
    whisper = Whisper()
    text = whisper.yt_transcribe(YT_URL, "transcribe")

    # with open("captions/transcript.txt", "w") as f:
    #     f.write(text)

    with open("captions/transcript.json", "w") as f:
        json.dump(text, f, indent=2)

if __name__ == "__main__":
    main()