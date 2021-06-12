"""
Multi-processing (多處理程序/多進程)：
    資料在彼此間傳遞變得更加複雜及花時間，因為一個 process 在作業系統的管理下是無法去存取別的 process 的 memory
    適合需要 CPU 密集，像是迴圈計算
Multi-threading (多執行緒/多線程)：
    資料彼此傳遞簡單，因為多執行緒的 memory 之間是共用的，但也因此要避免會有 Race Condition 問題
    適合需要 I/O 密集，像是爬蟲需要時間等待 request 回覆
"""
import logging
import threading
import time
from datetime import datetime

from submodule.Xu3.utils import getLogger


def main(urls, num):
    d = {"className": "Multi-processing and Multi-threading"}
    logger = getLogger(logger_name=datetime.now().strftime("%Y-%m-%d %H-%M-%S"),
                       logger_level=logging.DEBUG,
                       to_file=True,
                       time_file=False,
                       file_dir="test",
                       instance=True)

    logger.debug(f'開始執行({num})', extra=d)
    for url in urls:
        logger.debug(url, extra=d)
        time.sleep(1)
    logger.debug(f'結束({num})', extra=d)


url_list1 = [11, 12, 13, 14]
url_list2 = [21, 22]
url_list3 = [31, 32, 33]

# 定義線程
t1 = threading.Thread(target=main, args=(url_list1, 1))
t2 = threading.Thread(target=main, args=(url_list2, 2))
t3 = threading.Thread(target=main, args=(url_list3, 3))
t_list = [t1, t2, t3]

# 開始工作
for t in t_list:
    t.start()

# # 定義線程
# p1 = mp.Process(target=main, args=(url_list1, 2))
# p2 = mp.Process(target=main, args=(url_list2, 2))
# p3 = mp.Process(target=main, args=(url_list3, 2))
# p_list = [p1, p2, p3]
#
# if __name__ == "__main__":
#     print("本機 Process 可用最大數量:", mp.cpu_count())
#
#     # 開始工作
#     for p in p_list:
#         p.start()
