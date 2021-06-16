from abc import ABCMeta, abstractmethod


class Environment(metaclass=ABCMeta):
    def __init__(self):
        pass

    @abstractmethod
    def modifyEarlyStop(self):
        # TODO: 定義當生命體無法在獲得更好表現時的提早停止演化機制
        # TODO: 環境應可"部分影響"生命體的變異機率(表現不夠好時，促進變異；表現足夠好時，則減緩變異、甚至歸零)，主要影響細胞結構
        pass

    # 評估表現
    @abstractmethod
    def assessPerformance(self, outputs):
        # TODO: 評估生命體的表現，根據表現給予能量
        pass

    # 演化
    @abstractmethod
    def evolve(self, n_generation):
        # TODO: 每次演化都應給予不同的考驗
        # TODO: for yield 的形式，主要的演化過程定義在 evolve 當中，但可透過 yield 使外部可呈現現在演化的狀態
        for gen in range(n_generation):
            yield None
