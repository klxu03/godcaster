from module import RoundSplitter
from load_distributor import load_distribute
import sys

def main():
    index = int(sys.argv[1]) # 0-indexed
    total = int(sys.argv[2])
    videos_to_split = load_distribute(total, index)

    splitter = RoundSplitter("1_39.jpg", "/scratch/kxu39/merged_old/subsample/")
    
    for video in videos_to_split:
        splitter.split(video)

if __name__ == "__main__":
    main()