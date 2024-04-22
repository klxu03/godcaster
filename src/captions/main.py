from WhisperXRunner import WhisperXRunner
from load_distributor import load_distribute
import sys

def main():
    index = int(sys.argv[1]) # 0-indexed
    max_index = int(sys.argv[2])
    videos_to_split = load_distribute(max_index, index)

    # print("videos_to_split", videos_to_split)
    for video in videos_to_split:
        runner = WhisperXRunner(f"{video}/", f"{video}/", model_name="large-v3", compute_type="float16", batch_size=16)
        runner.run()

if __name__ == "__main__":
    main()