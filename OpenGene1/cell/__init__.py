import datetime
import math
from abc import ABCMeta, abstractmethod

import numpy as np

from OpenGene1.gene import Gene, createGene
from submodule.Xu3.utils import getLogger


class Cell(metaclass=ABCMeta):
    def __init__(self, logger_dir="cell", logger_name=datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")):
        self.logger_dir = logger_dir
        self.logger_name = logger_name
        self.extra = {"className": self.__class__.__name__}
        self.logger = getLogger(logger_name=self.logger_name,
                                to_file=False,
                                time_file=False,
                                file_dir=self.logger_dir,
                                instance=True)

    @abstractmethod
    def call(self, input_data):
        pass


class BaseCell(Cell, metaclass=ABCMeta):
    """
    Cell 如何解讀傳入的基因段，可以根據不同類型的 Cell 有不同的定義。

    @abstractmethod -> 定義要子物件實作的函式
    """
    # 定義結構的基因組個數
    n_struct = None

    # 定義數值的基因組個數
    n_value = None

    # 定義每個細胞所使用的基因數量(包含前 16 位基因用於區分種類)
    n_gene = 4096

    # 前 16 位基因用於區分細胞種類
    n_code = 16

    def __init__(self, gene, n_struct, n_value,
                 logger_dir="cell", logger_name=datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")):
        """

        :param gene: 傳入的基因段(不包含定義種類的基因，因此長度應為 4080 個)
        :param n_struct: 定義結構的基因組個數
        :param n_value: 定義數值的基因組個數
        """
        super().__init__(logger_dir=logger_dir, logger_name=logger_name)

        self.index = 0
        self.gene = gene

        self.__class__.n_struct = n_struct
        self.__class__.n_value = n_value

    @staticmethod
    def createCellGene():
        return createGene(n_gene=BaseCell.n_gene - BaseCell.n_code)

    @classmethod
    def getGeneDemand(cls):
        """
        不同類型的細胞，會需要不同的 n_struct 和 n_value。


        :return: 建構這個細胞所需的基因數量
        """
        gene_demand = cls.n_struct * Gene.struct_digits + (cls.n_value - 1) * Gene.value_steps + Gene.value_digits

        return gene_demand

    @abstractmethod
    def build(self):
        pass

    @abstractmethod
    def call(self, input_data):
        pass

    def nextStructGenome(self):
        for _ in range(self.__class__.n_struct):
            genome = self.gene[self.index: self.index + Gene.struct_digits]
            self.index += Gene.struct_digits

            yield genome

    def nextValueGenome(self):
        """

        :return: 返回長度為 Gene.digits 的基因組
        """
        for _ in range(self.__class__.n_value):
            genome = self.gene[self.index: self.index + Gene.value_digits]
            self.index += Gene.value_steps

            yield genome


class RawCell(Cell):
    def __init__(self, logger_dir="raw_cell", logger_name=datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")):
        super().__init__(logger_dir=logger_dir, logger_name=logger_name)

    def call(self, input_data):
        # input_data 在輸入前，會將多個輸入根據各維度最大值來做合併，因此就算從多個節點獲得數據，輸入也只會有一個
        return input_data


class ArbitraryCell(Cell):
    def __init__(self, cell: np.array,
                 logger_dir="arbitrary_cell", logger_name=datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")):
        super().__init__(logger_dir=logger_dir, logger_name=logger_name)
        self.cell = cell

    def build(self):
        pass

    def call(self, input_data):
        """
        輸入數據'維度較大'的軸，將 self.cell 暫時擴充整數倍 math.ceil(i / c)，到大於或等於輸入數據的大小。

        輸入數據'維度較小'的軸，填充 1 到與 self.cell 大小相同，由於數值是 1，和 self.cell 相乘後仍是 self.cell 的數值。


        :param input_data:
        :return:
        """
        cell = self.cell.copy()
        c, h, w = cell.shape
        ci, hi, wi = input_data.shape

        # cell 的各維度放大倍率
        sc, sh, sw = 1, 1, 1

        # region 確保 cell 尺寸大於或等於 input_data
        if ci > c:
            sc = math.ceil(ci / c)

        if hi > h:
            sh = math.ceil(hi / h)

        if wi > w:
            sw = math.ceil(wi / w)

        # 此處的 c, h, w 是
        self.logger.debug(f"(sc, sh, sw) = ({sc}, {sh}, {sw})", extra=self.extra)
        cell = np.tile(cell, (sc, sh, sw))
        # endregion

        c, h, w = cell.shape
        self.logger.debug(f"New (c, h, w) = ({c}, {h}, {w})", extra=self.extra)

        # 考慮 self.cell, x 形狀不同且無法 Broadcast 的情況
        x = np.pad(input_data,
                   pad_width=((0, c - ci), (0, h - hi), (0, w - wi)),
                   mode='constant',
                   constant_values=1)

        ci, hi, wi = x.shape
        self.logger.debug(f"New (ci, hi, wi) = ({ci}, {hi}, {wi})", extra=self.extra)

        # x = x[:c, :h, :w]
        output = cell * x

        return output


if __name__ == "__main__":
    def testCell():
        gene = createGene(n_gene=24)
        cell = BaseCell(gene, n_struct=2, n_value=3)

        struct_genome = cell.nextStructGenome()
        print("nextStructGenome:", next(struct_genome))
        print("nextStructGenome:", next(struct_genome))

        value_genome = cell.nextValueGenome()
        print("nextValueGenome:", next(value_genome))
        print("nextValueGenome:", next(value_genome))
        print("nextValueGenome:", next(value_genome))


    def testArbitraryCell():
        x = np.random.rand(1, 5, 4)
        cell = np.random.rand(2, 3, 2)
        ac = ArbitraryCell(cell=cell)
        output = ac.call(input_data=x)
        print(output.shape)


    # testCell()
    testArbitraryCell()
