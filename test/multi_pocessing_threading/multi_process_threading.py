import multiprocessing as mp
import threading
import time
from queue import Queue

import numpy as np

from submodule.events import Event


def threadingFunc(dictionary: dict):
    array = dictionary["array"]
    process_idx = dictionary["process_idx"]
    thread_idx = dictionary["thread_idx"]
    queue = dictionary["queue"]
    share_memory = dictionary["share_memory"]
    onThreadingEnd = dictionary["onThreadingEnd"]

    for i in array:
        queue.put(i)
        share_memory.addOhlc(i)
        print(f"{process_idx}-{thread_idx} | {share_memory}")
        time.sleep(1)

    onThreadingEnd(process_idx, thread_idx)


def processThreading(dictionary: dict):
    def onThreadingEndListener(idx, num):
        print(f"Thread ({idx}-{num}) end.")

    matrix = dictionary["matrix"]
    threadingFunc = dictionary["threadingFunc"]
    process_idx = dictionary["process_idx"]
    queue = Queue()
    total = 0
    event = Event()
    event.onThreadingEnd += onThreadingEndListener
    share_memory = []

    # 定義線程
    n_array = len(matrix)
    thread_list = []

    for i in range(n_array):
        thread = threading.Thread(target=threadingFunc, args=(dict(array=matrix[i],
                                                                   process_idx=process_idx,
                                                                   queue=queue,
                                                                   onThreadingEnd=event.onThreadingEnd,
                                                                   share_memory=share_memory,
                                                                   thread_idx=i + 1),))
        thread_list.append(thread)

    # 開始工作
    for t in thread_list:
        t.start()

    for t in thread_list:
        t.join()

    while not queue.empty():
        q = queue.get()
        total += q

    print(f"({process_idx}) total = {total}")


if __name__ == "__main__":
    print("本機 Process 可用最大數量:", mp.cpu_count())

    matrixs = np.arange(0, 9 * 3, 1).reshape((3, 3, -1))

    p1 = mp.Process(target=processThreading, args=(dict(matrix=matrixs[0],
                                                        threadingFunc=threadingFunc,
                                                        process_idx=1),))
    p2 = mp.Process(target=processThreading, args=(dict(matrix=matrixs[1],
                                                        threadingFunc=threadingFunc,
                                                        process_idx=2),))
    p3 = mp.Process(target=processThreading, args=(dict(matrix=matrixs[2],
                                                        threadingFunc=threadingFunc,
                                                        process_idx=3),))
    p_list = [p1, p2, p3]

    # 開始工作
    for p in p_list:
        p.start()

    # Main Process 繼續執行自己的工作
    proc = mp.current_process()
    print("{}, PID: {}".format(proc.name, proc.pid))

    # 等待所有Process執行結束
    for p in p_list:
        p.join()

    print("All Done.")
