from frames.module import RoundSplitter
import os

def main():
    splitter = RoundSplitter("1_39.jpg")
    dir_list = os.listdir("/scratch/kxu39")
    print(dir_list)
    print(dir_list[0])

if __name__ == "__main__":
    main()