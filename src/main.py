from frames.module import RoundSplitter
from frames.load_balancer import load_balance
import sys

def main():
    index = int(sys.argv[1]) # 0-indexed
    total = int(sys.argv[2])
    videos_to_split = load_balance(total, index)

    splitter = RoundSplitter("frames/1_39.jpg", "data")
    
    for video in videos_to_split:
        splitter.split(video)

if __name__ == "__main__":
    main()