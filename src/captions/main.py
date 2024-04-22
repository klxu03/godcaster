from WhisperXRunner import WhisperXRunner
from load_distributor import load_distribute
import sys
import os.path
import pickle
from LoggingSubprocess import LoggingSubprocess

def main():
    index = int(sys.argv[1]) # 0-indexed
    max_index = int(sys.argv[2])

    if os.path.isfile("./distributed_load.pkl"):
        with open("distributed_load.pkl", "rb") as f:
            indexes = pickle.load(f)
        videos_to_split = indexes[index]
    else:
        print("Run load_distribute, couldn't find pkl")
        videos_to_split = load_distribute(max_index, index)

    print("videos_to_split", videos_to_split)
    for video in videos_to_split:
        command = f"poetry run python WhisperXRunner.py {video}"
        LoggingSubprocess(command, shell=False).start()

if __name__ == "__main__":
    main()