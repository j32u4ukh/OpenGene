from abc import ABCMeta, abstractmethod

import numpy as np

from gene import translateStruct


class Organism(metaclass=ABCMeta):
    # TODO: 或許可在不同能量存量下，開關不同細胞的功能，例如關閉較不重要的、耗費較多能量的細胞

    # cls.n_gene 為初始化時的基因個數要求，會隨著基因變異而有所不同，實際基因長度會定義在 self.n_gene
    n_gene = 904268

    # 定義每個細胞所使用的基因數量(包含前 8 位基因用於區分種類)
    n_gene_per_cell = 4096

    def __init__(self, gene: np.array, n_cell: int = -1, energy: float = 100.0, mutation_rate: float = 0.03):
        # 生命運作所需能量
        self.energy = energy

        # 變異機率
        self.mutation_rate = mutation_rate

        self.gene = gene
        self.n_gene = len(gene)
        self.n_cell = n_cell
        self.structure = None

    @staticmethod
    def createGene(n_gene):
        """
        所需序列長度為 n_gene，此函示反過來約束根據前面幾位結構基因要求長度不可超過 n_gene。
        長度符合約束後，再產生剩餘的所需基因。

        1.細胞個數 (0 ~ 8)
        n = 2^a * 3^b * 5^c * 7^d -> a, b, c, d 分別由 2 個基因定義(0 ~ 3) -> 共 8 個基因來定義
        檢查後面的基因個數是否足夠建構該數量的細胞，若不足，則該細胞無法成立。
        a, b, c, d 同樣用於定義 -> 檢查次數、檢查位置分布

        2.細胞結構 (0 ~ n^2 - 1)
        n^2 個細胞，利用 Structure 來解析
        第一版 OpenGene 先令 n = 210

        3.數值
        由 Cell 來解析數值如何被定義與使用，每個細胞有 4096 個基因，不和其他細胞重疊，
        細胞內取值也是每 8 個取一次，同樣不重疊(取消 step 機制)

        (0 ~ 15) 前 16 個基因定義細胞類型，共 2^16 = 65536 種
        (16 ~ 4097) 後面 4080 個基因由各個基因各決定幾個用於定義結構，幾個用於定義數值

        :param n_gene: 要求產生基因所需長度
        :return:
         n_cell: 細胞個數
         gene: 返回符合基因結構要求的基因序列
        """
        # 檢查基因序列長度時，僅需前面 8 位基因即可
        struct_gene = np.random.randint(low=0, high=2, size=8)
        n_cell = Organism.checkGeneNumber(struct_gene, n_gene)

        while n_cell < 0:
            struct_gene = np.random.randint(low=0, high=2, size=8)
            n_cell = Organism.checkGeneNumber(struct_gene, n_gene)

        # 確保基因序列長度符合結構要求後，再產生後面所需長度的基因即可
        gene = np.random.randint(low=0, high=2, size=n_gene - 8)
        gene = np.append(struct_gene, gene)

        return n_cell, gene

    @staticmethod
    def checkGeneNumber(struct_gene, n_gene):
        base2 = translateStruct(struct_gene[0: 2])
        base3 = translateStruct(struct_gene[2: 4])
        base5 = translateStruct(struct_gene[4: 6])
        base7 = translateStruct(struct_gene[6: 8])

        # 計算該架構所需基因數量
        base_power = np.power([2.0, 3.0, 5.0, 7.0],
                              [base2, base3, base5, base7])
        n_cell = np.cumproduct(base_power)[-1]
        n_gene_demand = n_cell * (n_cell + Organism.n_gene_per_cell) + 8

        if n_gene >= n_gene_demand:
            return n_cell
        else:
            return -1

    @abstractmethod
    def formStructure(self, gene):
        """
        根據定義結構的基因，連結各個細胞，產生尚未包含數值的結構，n^2 個細胞，利用 Structure 來解析。

        :return: "細胞結構", "尚未使用到的基因"
        """
        pass

    @abstractmethod
    def call(self):
        pass

    @abstractmethod
    def toParams(self):
        pass
