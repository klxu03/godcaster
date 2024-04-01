from captions.module import Whisper

YT_URL = "https://www.youtube.com/watch?v=r4qS0WyH2Lk" 

def main():
    # populate audio
    whisper = Whisper()
    text = whisper.yt_transcribe(YT_URL, "transcribe")

    with open("captions/transcript.txt", "w") as f:
        f.write(text)

if __name__ == "__main__":
    main()