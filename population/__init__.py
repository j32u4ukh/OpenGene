from abc import ABCMeta, abstractmethod
import numpy as np
import os


class Population(metaclass=ABCMeta):
    # TODO: 讀取紀錄檔，並建置生命體；若無紀錄檔，則協助初始化
    # TODO: 不同族群之間也可以交換基因，但交換規則應和族群內不同，因為不同族群的基因組合應差異較族群內大
    def __init__(self, env: str, label: str, n_population: int = 500):
        self.env = env
        self.label = label

        path = f"data/{env}/{self.__class__}_{label}.txt"
        data = self.load(path=path)

        # 若記錄檔存在
        if os.path.exists(path):
            self.n_population = data["n_population"]
            self.organisms = data["organisms"]

        # 若記錄檔不存在
        else:
            # TODO: 若有數據黨，則由數據檔讀入參數，否則由 load 協助初始化
            self.n_population = n_population
            self.organisms = data["organisms"]

    def eliminate(self):
        # TODO: 檢查族群內生命體的能量存量，淘汰能量不足的生命體
        pass

    # 呼叫族群內所有生物的 call
    def call(self, input_datas: np.array) -> np.array:
        outputs = []

        for input_data, organism in zip(input_datas, self.organisms):
            output = organism.call(input_data)
            outputs.append(output)

        return np.array(outputs)

    @abstractmethod
    def load(self, path) -> dict:
        """
        根據檔案路徑(環境名稱-族群名稱-生命體名稱-label)，載入生命體的記錄檔
        根據不同 Population 定義的 n_population，以及對應的 Organism 所定義的生命體來初始化數據。

        :param path: 檔案路徑
        :return:
        """
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
