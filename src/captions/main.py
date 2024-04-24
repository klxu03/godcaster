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

def worker():
    while True:
        try:
            item = q.get()
        except queue.Queue.Empty:
            break

        try:
            print(f'Working on {item}')
            runner.run(item, item)
            print(f'Finished {item}')
        except:
            print("Runner run failed, retrying item", item)
            q.put(item)
            pass
        finally:
            q.task_done()

    q.task_done()

def main():
    index = int(sys.argv[1]) # 0-indexed
    max_index = int(sys.argv[2])

    multithread = True

    if os.path.isfile("./invalid_directories.txt"):
        with open("invalid_directories.txt", "r") as f:
            invalid_directories = f.read().split("\n")
        
        videos_to_split = [f"/scratch/kxu39/merged/{i}" for i in invalid_directories if i.strip()] 
        multithread = False
    elif os.path.isfile("./distributed_load.pkl"):
        with open("distributed_load.pkl", "rb") as f:
            indexes = pickle.load(f)
        videos_to_split = indexes[index]
    else:
        print("Run load_distribute, couldn't find pkl")
        videos_to_split = load_distribute(max_index, index)

    if multithread:
        NUM_THREADS = 8
        for _ in range(NUM_THREADS):
            threading.Thread(target=worker, daemon=True).start()

        print("videos_to_split", videos_to_split)
        for video in videos_to_split:
            q.put(f"{video}/")

        q.join()
    else:
        for video in videos_to_split:
            runner.run(video, video)

    print("Finished index", index)


if __name__ == "__main__":
    main()