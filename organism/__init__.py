from abc import ABCMeta, abstractmethod

from gene import createGene, Gene, translateStruct


class Organism(metaclass=ABCMeta):
    # TODO: 或許可在不同能量存量下，開關不同細胞的功能，例如關閉較不重要的、耗費較多能量的細胞
    n_gene = 904268

    def __init__(self, n_gene: int = 904268, energy: float = 100.0, mutation_rate: float = 0.03):
        # 生命運作所需能量
        self.energy = energy

        # 變異機率
        self.mutation_rate = mutation_rate

        self.n_gene = n_gene

        # 基因
        self.gene = createGene(n_gene=self.n_gene)

        """
        
        """

    @classmethod
    def checkGeneNumber(cls, gene):
        """
        根據前面幾位基因所定義的結構，初步檢查該細胞組合是否成立。
        成立: 返回"細胞結構"和"用於定義數值的基因"

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

        :return: "細胞結構", "尚未使用到的基因"
        """
        base2 = translateStruct(gene[0: 2])
        base3 = translateStruct(gene[2: 4])
        base5 = translateStruct(gene[4: 6])
        base7 = translateStruct(gene[6: 8])

        # 計算該架構所需基因數量
        n_cell = 2 ** base2 * 3 ** base3 * 5 ** base5 * 7 ** base7
        n_gene_demand = 8 + n_cell ** 2 + 4096 * n_cell

        # 檢查基因數量是否足夠
        if cls.n_gene >= n_gene_demand:
            return n_cell, gene[8:]
        else:
            return None, None

    @classmethod
    @abstractmethod
    def parseGene(cls, gene):
        """
        根據前面幾位基因所定義的結構，初步檢查該細胞組合是否成立。
        成立: 返回"細胞結構"和"用於定義數值的基因"

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

        :return: "細胞結構", "尚未使用到的基因"
        """
        base2 = translateStruct(gene[0: 2])
        base3 = translateStruct(gene[2: 4])
        base5 = translateStruct(gene[4: 6])
        base7 = translateStruct(gene[6: 8])

        # 計算該架構所需基因數量
        n_cell = 2**base2 * 3**base3 * 5**base5 * 7**base7
        n_gene_demand = 8 + n_cell**2 + 4096 * n_cell

        # 檢查基因數量是否足夠
        if cls.n_gene >= n_gene_demand:
            return n_cell, gene[8:]
        else:
            return None, None

    @abstractmethod
    def call(self):
        pass
