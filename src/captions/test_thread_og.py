import time
import queue
import threading

q = queue.Queue()

def worker():
    while True:
        try: 
            item = q.get()
        except queue.Queue.Empty:
            print("Queue empty")
            break

        print("Going to sleep..", item)
        time.sleep(item)
        print("Slept ..")
        q.task_done()


for j in range(3):
    threading.Thread(target=worker, daemon=True).start()


for i in range(3):
    q.put(i)

q.join()

print("done!!")