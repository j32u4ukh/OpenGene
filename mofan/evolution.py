import datetime
from abc import ABCMeta, abstractmethod

import numpy as np

# from submodule.Xu3.utils import getLogger
# TODO: OpenGene 應該要像 PyTorch 那樣，可以給其他人繼承，來達到最基本的機能架構


class Evolution(metaclass=ABCMeta):
    """
    1. 根據適應度相對高低為機率，抽取出基因組來進行繁殖
    2. 以 self.n_population * self.reproduction_rate 來決定每次繁殖的數量；1 / self.reproduction_rate 來決定繁衍幾倍子代
    3. 子代加入族群中，與親代共同被進行排序，表現差的基因組則剔除
    """
    def __init__(self, n_population, logger_dir="backtest", logger_name=datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")):
        # self.logger_dir = logger_dir
        # self.logger_name = logger_name
        # self.extra = {"className": self.__class__.__name__}
        # self.logger = getLogger(logger_name=self.logger_name,
        #                         to_file=True,
        #                         time_file=False,
        #                         file_dir=self.logger_dir,
        #                         instance=True)

        # 族群個數: 可變動
        self.n_population = n_population

        # 族群個數基數: 幾乎不變動
        self.N_POPULATION = n_population

        # 族群整體
        self.population = None

        # 初始化族群
        self.initPopulation()

        """
        族群的大小，除了會受到物理或生物因子的調控之外，生物本身的生物潛能（biotic potential）與
        環境的負荷量（carrying capacity），也會影響到族群的大小。
        對生物而言，族群的個體數至少要多少，才能維持族群不致於滅絕呢？
        """
        # 繁殖倍率，種族數量越少，倍率越高(或應限制不高於 0.5，以該數值劃分好基因和壞基因)
        self.reproduction_rate = 0.25
        self.reproduction_scale = min(10, int(1 / self.reproduction_rate))
        self.early_stop = 1e-5

        self.fitness = 0
        self.POTENTIAL = 5
        self.potential = 5

    # 初始化族群
    @abstractmethod
    def initPopulation(self):
        pass

    def setPotential(self, potential=5):
        """
        適應度無法再提升的情況下，應再嘗試演化幾輪的定義。
        :param potential: 演化幾輪
        :return:
        """
        self.POTENTIAL = potential
        self.potential = potential

    def resetPotential(self):
        """
        之前在適應度無法再提升的情況下，隨著嘗試演化數輪，適應度再次提升，因此須將嘗試次數還原，
        以便下次出現適應度無法再提升的情況時，可以再次使用。
        :return:
        """
        self.potential = self.POTENTIAL

    def updatePopulationSize(self, n_population):
        # TODO: 環境提供能量越多，可支持的族群數量越大
        self.n_population = n_population

        # 以 0.25 為中間值，值域為 0 ~ 0.5 的 sigmoid 函數。實際數量小於應有數量時，繁殖倍率會上升；反之則下降。
        self.reproduction_rate = 0.5 / (1 + np.power(np.e, self.N_POPULATION / self.n_population))

        self.reproduction_scale = min(10, int(1 / self.reproduction_rate))

    # RNA 轉譯
    @abstractmethod
    def translation(self):
        pass

    @abstractmethod
    def evolve(self, *args, **kwargs):
        """
        封裝 '''計算適應度 -> 繁殖 -> 計算適應度 -> 淘汰''' 的過程
        :param args:
        :param kwargs:
        :return:
        """
        pass

    # 計算環境適應度並排序
    @abstractmethod
    def getFitness(self, *args, **kwargs):
        pass

    # 繁殖
    @abstractmethod
    def reproduction(self, *args, **kwargs):
        """
        定義哪些基因組來進行繁殖，以及每次繁殖多少子代。
        定義如何重組基因組來源們和自身的基因。
        :return: 子代
        """
        pass

    # 基因交換
    @abstractmethod
    def geneExchange(self, *args, **kwargs):
        pass

    # 基因變異
    @abstractmethod
    def mutate(self, *args, **kwargs):
        pass

    def addOffspring(self, offspring):
        # 加入族群中
        # np.append 返回添加後的結果，不改變原始陣列，將陣列 2 加到陣列 1 當中，沿著指定的 axis
        self.population = np.append(self.population, offspring, axis=0)
        self.updatePopulationSize(n_population=len(self.population))

    # 不再進步時，提前終止演化
    @abstractmethod
    def modifyEarlyStop(self, *args, **kwargs):
        pass

    # 天擇
    @abstractmethod
    def naturalSelection(self, *args, **kwargs):
        """
        定義淘汰機制。
        :param args:
        :param kwargs:
        :return:
        """
        pass


class FragmentEvolution(Evolution):
    """
    1. 根據適應度相對高低為機率，抽取出基因組來進行繁殖
    2. 以 self.n_population * self.reproduction_rate 來決定每次繁殖的數量；1 / self.reproduction_rate 來決定繁衍幾倍子代
    3. 子代加入族群中，與親代共同被進行排序，表現差的基因組則剔除
    """
    def __init__(self, n_population, fragment_size, n_fragment=1, logger_name="FragmentEvolution"):

        # rna 片段，一到多組 rna 片段組成完整的 rna
        self.fragment_size = fragment_size
        self.n_fragment = n_fragment

        # rna 長度: 暫時不可變動
        self.rna_size = fragment_size * n_fragment

        # dna.shape: (fragment_size * n_fragment, 2)(2: mu & std 為一對基因)
        # [[mu1,  mu2,  ..., mu_n],
        #  [std1, std2, ..., std_n]]
        self.dna_size = self.rna_size * 2

        # 變異強度
        self.mutation_strength = 1.0
        self.mutation_scale = 0.8

        super().__init__(n_population=n_population, logger_name=logger_name)

    # 初始化族群
    def initPopulation(self):
        pass

    # RNA 轉譯
    def translation(self):
        pass

    # 演化
    def evolve(self, *args, **kwargs):
        """
        封裝 '''計算適應度 -> 繁殖 -> 計算適應度 -> 淘汰''' 的過程
        :param args:
        :param kwargs:
        :return:
        """
        pass

    # 計算環境適應度並排序
    def getFitness(self, *args, **kwargs):
        pass

    # 繁殖
    def reproduction(self, *args, **kwargs):
        """
        定義哪些基因組來進行繁殖，以及每次繁殖多少子代。
        定義如何重組基因組來源們和自身的基因。
        :return: 子代
        """
        pass

    # 基因交換
    def geneExchange(self, *args, **kwargs):
        pass

    # 基因變異
    def mutate(self, *args, **kwargs):
        pass

    # 不再進步時，提前終止演化
    def modifyEarlyStop(self, *args, **kwargs):
        pass

    # 天擇
    def naturalSelection(self, *args, **kwargs):
        """
        定義淘汰機制。
        :param args:
        :param kwargs:
        :return:
        """
        pass