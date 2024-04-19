from frames.module import RoundSplitter
from frames.load_balancer import load_balance
import sys

def main():
    index = sys.argv[1]
    total = sys.argv[2]
    videos_to_split = load_balance(total, index)

    splitter = RoundSplitter("1_39.jpg")

if __name__ == "__main__":
    main()