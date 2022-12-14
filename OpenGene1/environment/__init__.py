from abc import ABCMeta, abstractmethod


class Environment(metaclass=ABCMeta):
    def __init__(self):
        pass

    @abstractmethod
    def modifyEarlyStop(self):
        # TODO: 定義當生命體無法在獲得更好表現時的提早停止演化機制
        # TODO: 環境應可"部分影響"生命體的變異機率(表現不夠好時，促進變異；表現足夠好時，則減緩變異、甚至歸零)，主要影響細胞結構
        #  -> 並非直接影響或傳入參數來影響，而是族群在評估整體表現時，若較不好，則會有較高的結構基因變異。
        #  結構基因變異率為任務表現的函數，f(任務表現) = 結構基因變異率。
        pass

    # 評估表現
    @abstractmethod
    def assessPerformance(self, outputs):
        # TODO: 評估生命體的表現，根據表現給予能量(無論任務為何，一律返回 表現評價 & 能量回饋)
        # TODO: 同一個任務下，會有多個類似的環境，利用參數來調節最重視的項目，
        #  例如"表現正確率最重要，計算時間就算長也沒關係"或"表現一定程度(比如 60 分)以上即可，更重視計算速度"
        #  -> 各個環境演化出最適應彼此的生命體後，會讓不同環境下的族群混種，交配產生出"表現好(80 分以上)，計算速度也快"的新生命體
        pass

    # 演化
    @abstractmethod
    def evolve(self, n_generation):
        # TODO: 每次演化都應給予不同的考驗
        # TODO: for yield 的形式，主要的演化過程定義在 evolve 當中，但可透過 yield 使外部可呈現現在演化的狀態
        for gen in range(n_generation):
            yield None
