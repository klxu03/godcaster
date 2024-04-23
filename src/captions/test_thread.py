import time
import queue
import threading

class worker(threading.Thread):

    def __init__(self,qu):
        threading.Thread.__init__(self)
        self.que=qu

    def run(self):
        for item in iter(self.que.get, None): # This will call self.que.get() until None is returned, at which point the loop will break.
            print("Going to sleep..")
            time.sleep(item)
            print("Slept ..")
            self.que.task_done()
        self.que.task_done()
        print("Thread is done!")


q = queue.Queue()

for j in range(3):
    work = worker(q);
    work.setDaemon(True)
    work.start()


for i in range(5):
    q.put(1)

for i in range(3):  # Shut down all the workers
    q.put(None)

q.join()

print("done!!")