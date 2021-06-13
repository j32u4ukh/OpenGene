import multiprocessing as mp
import threading
import time


def threadingFunc(dictionary: dict, process_id=0, thread_id=0):
    value = dictionary["value"]
    result = 0

    for i in range(10):
        result += value
        print(f"{process_id}-{thread_id} | {result}")
        time.sleep(0.01)


def processThreading(dictionary: dict, process_id=0):
    params = dictionary["params"]

    # 定義線程
    thread_list = []

    for thread_id in range(3):
        print(f"process_id: {process_id}, thread_id: {thread_id}")
        thread = threading.Thread(target=threadingFunc, args=(dict(value=params[thread_id]["value"]),
                                                              process_id,
                                                              thread_id))
        thread_list.append(thread)

    # 開始工作
    for t in thread_list:
        t.start()

    for t in thread_list:
        t.join()


def run():
    dictionary = {}
    n_process = 3
    n_threading = 3

    for p in range(n_process):
        dictionary[p] = {}

        for t in range(n_threading):
            dictionary[p][t] = {"value": p + t + 1}

    print(dictionary)
    p_list = []

    for process_id in range(n_process):
        p = mp.Process(target=processThreading, args=(dict(params=dictionary[process_id]), process_id))
        p_list.append(p)

    # 開始工作
    for p in p_list:
        p.start()

    # 等待所有Process執行結束
    for p in p_list:
        p.join()

    print("All Done.")


if __name__ == "__main__":
    run()
