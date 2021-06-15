from abc import ABCMeta, abstractmethod

from gene import createGene


class Organism(metaclass=ABCMeta):
    # TODO: 或許可在不同能量存量下，開關不同細胞的功能，例如關閉較不重要的、耗費較多能量的細胞
    n_gene = 0

    def __init__(self, energy, mutation_rate):
        # 生命運作所需能量
        self.energy = energy

        # 變異機率
        self.mutation_rate = mutation_rate

        # 基因
        self.gene = createGene(n_gene=self.__class__.n_gene)

    def parseGene(self):
        # TODO: self.gene -> 1. 基本參數 2. 細胞結構 3. 數值
        pass

    @abstractmethod
    def call(self):
        pass
