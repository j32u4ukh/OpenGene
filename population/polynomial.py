from population import Population

"""
mode 1
# 根據適應度換算成機率，越高被選擇到的機率則越高，適應度低的基因組也有機會被選到，但機率也比較低
# 基因交換為根據被選到的索引值，進行單純交換

mode 2
# 強者基因不變，但會使用部分強者基因去替換弱者基因，替換位置為隨機

mode 3
# 基因定義了高斯分配的平均數和變異數(基因變異幅度)，

mode 4
# 子代比親代優秀 -> MUT_STRENGTH *= 大 -> 持續變異，探索可能空間 2.028114981647472
# 親代比子代優秀 -> MUT_STRENGTH *= 小 -> 變異收斂 0.8379668855787558
"""


# 一元一次方程式 f(x) = a * x + b
class UnaryLinearPopulation(Population):
    def __init__(self, env: str, label: str, n_population):
        super().__init__(env=env, label=label, n_population=n_population)

    def load(self, env: str, label: str):
        pass

    def reproduction(self):
        pass

    def geneExchange(self):
        pass
