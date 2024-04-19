from .module import RoundSplitter
import os

if __name__ == "__main__":
    splitter = RoundSplitter("1_39.jpg")
    dir_list = os.listdir("/scratch/kxu39")
    print(dir_list)

