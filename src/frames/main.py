from module import RoundSplitter
from load_distributor import load_distribute
import sys

def main():
    index = int(sys.argv[1]) # 0-indexed
    max_index = int(sys.argv[2])
    videos_to_split = load_distribute(max_index, index)

    splitter = RoundSplitter("1_39.jpg", "/scratch/kxu39/merged_old/subsample/")
    
    print("videos_to_split", videos_to_split)
    for video in videos_to_split:
        splitter.split(video)

if __name__ == "__main__":
    main()