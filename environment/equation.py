import numpy as np
from environment import Environment
from abc import ABCMeta, abstractmethod


class Equation(Environment, metaclass=ABCMeta):
    def __init__(self):
        super().__init__()

    @staticmethod
    def func(x):
        return (np.sin(10 * x) + np.cos(2 * x)) * x

    @abstractmethod
    def modifyEarlyStop(self):
        pass

    @abstractmethod
    def assessPerformance(self):
        pass

    @abstractmethod
    def evolve(self):
        pass


class SimpleExchangeEquation(Equation):
    def __init__(self):
        super().__init__()

    def modifyEarlyStop(self):
        # TODO: 定義當生命體無法在獲得更好表現時的提早停止演化機制
        pass

    # 評估表現
    def assessPerformance(self):
        # TODO: 評估生命體的表現，根據表現給予能量
        pass

    # 演化
    def evolve(self):
        # TODO: 每次演化都應給予不同的考驗
        pass


class WinnerLoserEquation(Equation):
    def __init__(self):
        super().__init__()

    def modifyEarlyStop(self):
        # TODO: 定義當生命體無法在獲得更好表現時的提早停止演化機制
        pass

    # 評估表現
    def assessPerformance(self):
        # TODO: 評估生命體的表現，根據表現給予能量
        pass

    # 演化
    def evolve(self):
        # TODO: 每次演化都應給予不同的考驗
        pass


class DistributionEquation(Equation):
    def __init__(self):
        super().__init__()

    def modifyEarlyStop(self):
        # TODO: 定義當生命體無法在獲得更好表現時的提早停止演化機制
        pass

    # 評估表現
    def assessPerformance(self):
        # TODO: 評估生命體的表現，根據表現給予能量
        pass

    # 演化
    def evolve(self):
        # TODO: 每次演化都應給予不同的考驗
        pass


class MutationStrengthDistributionEquation(Equation):
    def __init__(self):
        super().__init__()

    def modifyEarlyStop(self):
        # TODO: 定義當生命體無法在獲得更好表現時的提早停止演化機制
        pass

    # 評估表現
    def assessPerformance(self):
        # TODO: 評估生命體的表現，根據表現給予能量
        pass

    # 演化
    def evolve(self):
        # TODO: 每次演化都應給予不同的考驗
        pass
