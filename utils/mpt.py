import multiprocessing as mp
import threading
import time
from abc import ABCMeta, abstractmethod


class MPT(metaclass=ABCMeta):
    def __init__(self, n_process: int = 1, n_threading: int = 1):
        """
        當 n_process = 1 且 threading = 1，則直接執行目標函式。
        若 n_process = 1 但 threading > 1，則使用多執行序來執行目標函式。
        若 n_process > 1，則使用多核心來執行目標函式。

        :param n_process: 核心數
        :param n_threading: 執行序數
        """
        self.n_process = min(mp.cpu_count(), max(n_process, 1))
        self.n_threading = min(20, max(n_threading, 1))
        self.params = dict()

    @abstractmethod
    def threadingFunc(self, thread_params, process_id=0, thread_id=0):
        pass

    def processThreading(self, process_params, process_id=0):
        thread_list = []

        for thread_id in range(self.n_threading):
            thread = threading.Thread(target=self.threadingFunc,
                                      args=(process_params[thread_id], process_id, thread_id))
            thread_list.append(thread)

        # 開始工作
        for t in thread_list:
            t.start()

        for t in thread_list:
            t.join()

    def run(self):
        if self.n_process == 1:
            if self.n_threading == 1:
                self.threadingFunc(self.params[0][0])
            else:
                self.processThreading(self.params[0])
        else:
            process_list = []

            for process_id in range(self.n_process):
                process = mp.Process(target=self.processThreading, args=(self.params[process_id], process_id,))
                process_list.append(process)

            # 開始工作
            for p in process_list:
                p.start()

            for p in process_list:
                p.join()


class MptDemo(MPT):
    def __init__(self, n_process=2, n_threading=5):
        super().__init__(n_process=n_process, n_threading=n_threading)

    def threadingFunc(self, thread_params, process_id=0, thread_id=0):
        result = 0
        value = thread_params["value"]

        for i in range(10):
            result += value
            print(f"process_id: {process_id}, thread_id: {thread_id}, result: {result}")
            time.sleep(0.01)


if __name__ == "__main__":
    n_process = 2
    n_threading = 2
    mpt_demo = MptDemo(n_process=n_process, n_threading=n_threading)

    for p in range(n_process):
        mpt_demo.params[p] = {}

        for t in range(n_threading):
            mpt_demo.params[p][t] = {"value": p + t + 1}

    mpt_demo.run()
