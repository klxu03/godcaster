from WhisperXRunner import WhisperXRunner
from load_distributor import load_distribute
import sys
import os.path
import pickle

import threading
import queue
from dotenv import load_dotenv

q = queue.Queue()
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")
runner = WhisperXRunner(model_name="large-v3", compute_type="float16", batch_size=16, hf_token=HF_TOKEN)

class worker(threading.Thread):
    def __init__(self, q):
        threading.Thread.__init__(self)
        self.q = q

    def run(self):
        for item in iter(self.q.get, None):
            print(f'Working on {item}')
            runner.run(item, item)
            print(f'Finished {item}')
            self.q.task_done()
        self.q.task_done() # thread is done now

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

    NUM_THREADS = 3
    for _ in range(NUM_THREADS):
        work = worker(q)
        work.setDaemon(True)
        work.start()

    print("videos_to_split", videos_to_split)
    for video in videos_to_split:
        q.put(f"{video}/")

    q.join()
    print("Finished index", index)


if __name__ == "__main__":
    main()