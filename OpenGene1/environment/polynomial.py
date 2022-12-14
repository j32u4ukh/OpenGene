from OpenGene1.environment import Environment
from OpenGene1.population.polynomial import UnaryLinearPopulation
import numpy as np


# TODO: 先解決固定參數的一元一次方程式，再透過傳入多組解來求出任意參數一元一次方程式
# 待解決任務：一元一次方程式
class UnaryLinear(Environment):
    def __init__(self, n_population, equation, label="0"):
        super().__init__()
        self.population = UnaryLinearPopulation(env=self.__class__.__name__,
                                                label=label,
                                                n_population=n_population)
        self.equation = equation

    def modifyEarlyStop(self):
        pass

    # 評估表現
    def assessPerformance(self, outputs):
        pass

    # 演化
    def evolve(self, n_generation):
        input_datas = np.random.random((1, 3, 4))
        outputs = self.population.call(input_datas)
