# TODO: 或許可在不同能量存量下，開關不同細胞的功能，例如關閉較不重要的、耗費較多能量的細胞
from abc import ABCMeta, abstractmethod


class Organism(metaclass=ABCMeta):
    def __init__(self, mutation_rate):
        # 變異機率
        self.mutation_rate = mutation_rate

