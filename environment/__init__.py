from abc import ABCMeta, abstractmethod


class Environment(metaclass=ABCMeta):
    def __init__(self):
        # TODO: 定義同時存在多少族群(使用多少核心或不共通的執行序來計算)，每個族群使用多少執行序，每個執行序有多少個體
        pass

    @abstractmethod
    def modifyEarlyStop(self):
        # TODO: 定義當生命體無法在獲得更好表現時的提早停止演化機制
        pass

    # 評估表現
    @abstractmethod
    def assessPerformance(self):
        # TODO: 評估生命體的表現，根據表現給予能量
        pass

    # 演化
    @abstractmethod
    def evolve(self):
        # TODO: 每次演化都應給予不同的考驗
        pass
