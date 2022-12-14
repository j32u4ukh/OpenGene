from abc import ABCMeta, abstractmethod


class Evolution(metaclass=ABCMeta):
    def __init__(self, n_populations, n_family):
        # TODO: 定義同時存在多少族群(使用多少核心或不共通的執行序來計算)，每個族群使用多少執行序，每個執行序有多少個體

        # 多少個族群 -> 使用幾個核心
        self.n_populations = n_populations

        # 多少個家族 -> 每個核心使用多少執行序
        self.n_family = n_family

    # 開始進行演化
    @abstractmethod
    def startBigBang(self):
        # TODO: 或許應有個次級的開始演化，在每次暫停後呼叫，而非每次都從大爆炸開始
        pass

    # 暫停演化(所有環境內的族群都提出了提早停止演化，可修改環境參數或允許不同族群之間的交流後，再次開始演化)
    @abstractmethod
    def pauseBigBang(self):
        pass
