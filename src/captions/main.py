from WhisperXRunner import WhisperXRunner
from load_distributor import load_distribute
import sys
import os.path
import pickle

import threading
import queue

q = queue.Queue()
runner = WhisperXRunner(model_name="large-v3", compute_type="float16", batch_size=16)

def worker():
    while True:
        try:
            item = q.get()
        except queue.Queue.Empty:
            break

        print(f'Working on {item}')
        runner.run(item, item)
        print(f'Finished {item}')

        q.task_done()

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

    for _ in range(10):
        threading.Thread(target=worker, daemon=True).start()

    print("videos_to_split", videos_to_split)
    for video in videos_to_split:
        q.put(f"{video}/")

    q.join()
    print("Finished index", index)


if __name__ == "__main__":
    main()