from abc import ABCMeta, abstractmethod


class Population(metaclass=ABCMeta):
    # TODO: 讀取紀錄檔，並建置生命體；若無紀錄檔，則協助初始化
    # TODO: 不同族群之間也可以交換基因，但交換規則應和族群內不同，因為不同族群的基因組合應差異較族群內大
    def __init__(self, n_population):
        self.n_population = n_population

    def eliminate(self):
        # TODO: 檢查族群內生命體的能量存量，淘汰能量不足的生命體
        pass

    # 繁殖
    @abstractmethod
    def reproduction(self):
        # TODO: 定義如何根據生命體的適應度(能量存量/每次計算消耗的能量)，以及基因相似性，來決定交配對象
        pass

    # 基因交換
    @abstractmethod
    def geneExchange(self):
        pass

