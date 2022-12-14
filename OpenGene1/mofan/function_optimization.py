import matplotlib.pyplot as plt
import numpy as np

from OpenGene1.mofan.evolution import FragmentEvolution


def func(x):
    return (np.sin(10 * x) + np.cos(2 * x)) * x


class SimpleExchangeDemo(FragmentEvolution):
    """genetic-algorithm-basic.py"""

    def __init__(self, value_range, fragment_size, n_population, mutation_rate):
        super().__init__(fragment_size=fragment_size, n_population=n_population, logger_name="SimpleExchangeDemo")
        self.mutation_rate = mutation_rate
        self.value_range = value_range
        # np.arange(self.rna_size): 產生 [0, 1, ..., self.rna_size - 1]
        self.translation_dot = 2 ** np.arange(self.rna_size)[::-1]

        # 1 + 2 + 4 + ... = 2^n -1
        value_range_size = self.value_range[1] - self.value_range[0]
        self.translation_multiplier = value_range_size / float(2 ** self.rna_size - 1)

    def initPopulation(self):
        # self.population.shape = (self.n_population, self.rna_size)
        # self.population = [[0, 1, 1, 0, ...], ..., [1, 1, 0, 0, ...]]
        self.population = np.random.randint(2, size=(self.n_population, self.rna_size))

    def translation(self):
        return self.population.dot(self.translation_dot) * self.translation_multiplier

    def evolve(self, *args, **kwargs):
        # 計算適應度並排序
        self.getFitness()

        # 繁殖
        self.reproduction()

        # 計算適應度並排序
        fitness = self.getFitness()

        self.modifyEarlyStop(fitness=fitness)

        # 淘汰機制
        self.naturalSelection()

    def getFitness(self, early_stop=False):
        # x = self.translation()
        # bound_limit = (self.value_range[0] <= x) & (x <= self.value_range[1])
        #
        # # 淘汰超出值域範圍的基因組
        # self.population = self.population[bound_limit]
        # self.updatePopulationSize(n_population=len(self.population))

        # 計算適應度
        fitness = func(self.translation())

        # 根據 values 數值大小，給予排名的數列，數值越小，排名數值越小
        # np.argsort([9, 4, 6]) -> array([1, 2, 0], dtype=int64)
        index = np.argsort(fitness)

        # 根據適應度排序: 讓最小的排最前面，最大的排最後面
        self.population = self.population[index]

        return fitness

    def reproduction(self, *args, **kwargs):
        # 取出最優秀的一批基因組
        n_best = int(self.N_POPULATION * self.reproduction_rate)
        n_best = min(n_best, self.n_population)

        # 根據適應度換算成機率，越高被選擇到的機率則越高，適應度低的基因組也有機會被選到，但機率也比較低
        array = np.arange(self.n_population)
        p = array / array.sum()
        parent1 = np.random.choice(array,
                                   size=n_best,
                                   replace=False,
                                   p=p)

        parent2 = np.random.choice(array,
                                   size=n_best,
                                   replace=False,
                                   p=p)

        children = self.geneExchange(self.population[parent1], self.population[parent2])

        # 產生變異
        children = self.mutate(children=children)

        # 加入族群中
        self.addOffspring(offspring=children)

    def geneExchange(self, *args):
        children = args[0].copy()
        other = args[1]

        # exchange_idx = [True,  True, False, ..., False, False]
        exchange_idx = np.random.randint(0, 2, size=(len(children), self.rna_size)).astype(np.bool)
        children[exchange_idx] = other[exchange_idx]

        for _ in range(self.reproduction_scale - 1):
            child = args[0].copy()
            exchange_idx = np.random.randint(0, 2, size=(len(child), self.rna_size)).astype(np.bool)
            child[exchange_idx] = other[exchange_idx]
            children = np.append(children, child, axis=0)

        return children

    def mutate(self, children: np.array):
        mutatation = np.random.random(children.shape) < self.mutation_rate
        mutatation_index0 = np.where(mutatation & (children == 0))
        mutatation_index1 = np.where(mutatation & (children == 1))

        children[mutatation_index0] = 1
        children[mutatation_index1] = 0

        return children

    def modifyEarlyStop(self, *args, **kwargs):
        average_fitness = kwargs['fitness'].mean()

        if average_fitness > self.fitness:
            self.fitness = average_fitness

            if self.potential < self.POTENTIAL:
                self.resetPotential()
        else:
            self.potential -= 1

    def naturalSelection(self, *args, **kwargs):
        """
        只保留較佳的基因組

        :param args:
        :param kwargs:
        :return:
        """
        self.population = self.population[-self.N_POPULATION:]
        self.updatePopulationSize(n_population=len(self.population))
        # self.logger.debug(f"n_population: {self.n_population}")


class WinnerLoserDemo(FragmentEvolution):
    """MicrobialGeneticAlgorithm.py"""

    def __init__(self, value_range, fragment_size, n_population, exchange_rate, mutation_rate):
        super().__init__(fragment_size=fragment_size, n_population=n_population, logger_name="WinnerLoserDemo")
        self.mutation_rate = mutation_rate
        self.value_range = value_range
        # np.arange(self.rna_size): 產生 [0, 1, ..., self.rna_size - 1]
        self.translation_dot = 2 ** np.arange(self.rna_size)[::-1]

        value_range_size = value_range[1] - value_range[0]
        self.translation_multiplier = value_range_size / float(2 ** self.rna_size - 1)

        self.exchange_rate = exchange_rate

    def initPopulation(self):
        self.population = np.random.randint(2, size=(self.n_population, self.rna_size))

    def translation(self):
        return self.population.dot(self.translation_dot) * self.translation_multiplier

    def evolve(self, *args, **kwargs):
        # 計算適應度並排序
        self.getFitness()

        # 繁殖
        self.reproduction()

        # 計算適應度並排序
        fitness = self.getFitness()

        self.modifyEarlyStop(fitness=fitness)

        # 淘汰機制
        self.naturalSelection()

    def getFitness(self, *args, **kwargs):
        # x = self.translation()
        # bound_limit = (self.value_range[0] <= x) & (x <= self.value_range[1])
        #
        # # 淘汰超出值域範圍的基因組
        # self.population = self.population[bound_limit]
        # self.updatePopulationSize(n_population=len(self.population))

        # 計算適應度
        values = func(self.translation())

        # 根據 values 數值大小，給予排名的數列，數值越小，排名數值越小
        # np.argsort([9, 4, 6]) -> array([1, 2, 0], dtype=int64)
        fitness = np.argsort(values)

        # 根據適應度排序: 讓最小的排最前面，最大的排最後面
        self.population = self.population[fitness]

        return values

    def reproduction(self, *args, **kwargs):
        n_reproduction = int(self.N_POPULATION * self.reproduction_rate)
        n_reproduction = min(n_reproduction, int(self.n_population / 2))

        # 根據適應度換算成機率，越高被選擇到的機率則越高，適應度低的基因組也有機會被選到，但機率也比較低
        array = np.arange(self.n_population)
        winner = np.random.choice(array,
                                  size=n_reproduction,
                                  replace=False,
                                  p=array / array.sum())

        array = array[::-1]
        loser = np.random.choice(array,
                                 size=n_reproduction,
                                 replace=False,
                                 p=array / array.sum())

        # 基因交換
        children = self.geneExchange(winner=self.population[winner], loser=self.population[loser])

        # 產生變異
        children = self.mutate(children=children)

        # 加入族群中
        self.addOffspring(offspring=children)
        # self.logger.debug(f"#population: {self.n_population}")

    def geneExchange(self, *args, **kwargs):
        children = kwargs["loser"].copy()
        flop = np.random.random(children.shape) < self.exchange_rate
        children[flop] = kwargs["winner"][flop]

        for _ in range(self.reproduction_scale - 1):
            child = kwargs["loser"].copy()
            flop = np.random.random(child.shape) < self.exchange_rate
            child[flop] = kwargs["winner"][flop]
            children = np.append(children, child, axis=0)

        return children

    def mutate(self, children: np.array):
        mutatation = np.random.random(children.shape) < self.mutation_rate
        mutatation_index0 = np.where(mutatation & (children == 0))
        mutatation_index1 = np.where(mutatation & (children == 1))

        children[mutatation_index0] = 1
        children[mutatation_index1] = 0

        return children

    def modifyEarlyStop(self, *args, **kwargs):
        average_fitness = kwargs['fitness'].mean()

        if average_fitness > self.fitness:
            self.fitness = average_fitness

            if self.potential < self.POTENTIAL:
                self.resetPotential()
        else:
            self.potential -= 1

    def naturalSelection(self, *args, **kwargs):
        self.population = self.population[-self.N_POPULATION:]
        self.updatePopulationSize(n_population=len(self.population))
        # self.logger.debug(f"n_population: {self.n_population}")


class DistributionDemo(FragmentEvolution):
    """EvolutionStrategyBasic.py"""

    def __init__(self, value_range, fragment_size, n_population, exchange_rate):
        self.value_range = value_range
        self.exchange_rate = exchange_rate
        self.mutation_strength = 2.0
        super().__init__(fragment_size=fragment_size, n_population=n_population, logger_name="DistributionDemo")

    def initPopulation(self):
        value_range = self.value_range[1] - self.value_range[0]

        mu = np.random.rand(self.n_population, self.rna_size) * value_range / self.rna_size
        std = np.random.rand(self.n_population, self.rna_size) * self.mutation_strength + 1e-5
        self.population = np.hstack((mu, std))

    def translation(self):
        mu = self.getMu()

        return mu.sum(axis=1)

    def evolve(self, *args, **kwargs):
        # 計算適應度並排序
        self.getFitness()

        # 繁殖
        self.reproduction()

        # 計算適應度並排序
        fitness = self.getFitness()

        self.modifyEarlyStop(fitness=fitness)

        # 淘汰機制
        self.naturalSelection()

    def getFitness(self, *args, **kwargs):
        mu = self.translation()
        valid_mu = np.where((self.value_range[0] <= mu) & (mu <= self.value_range[1]))
        # valid_std = self.getValidStdIndex()
        # valid_idx = np.intersect1d(valid_mu, valid_std)
        valid_idx = valid_mu

        # 淘汰超出值域範圍的基因組
        self.population = self.population[valid_idx]
        self.updatePopulationSize(n_population=len(self.population))
        # self.logger.info(f"#population: {self.n_population}")

        # 計算適應度
        fitness = func(self.translation())

        # 根據 values 數值大小，給予排名的數列，數值越小，排名數值越小
        # np.argsort([9, 4, 6]) -> array([1, 2, 0], dtype=int64)
        indexs = np.argsort(fitness)

        # 根據適應度排序: 讓最小的排最前面，最大的排最後面
        self.population = self.population[indexs]

        return fitness

    def reproduction(self, *args, **kwargs):
        n_reproduction = int(self.n_population * self.reproduction_rate)

        # 根據適應度換算成機率，越高被選擇到的機率則越高，適應度低的基因組也有機會被選到，但機率也比較低
        array = np.arange(self.n_population)
        winner_idx = np.random.choice(array,
                                      size=n_reproduction,
                                      replace=False,
                                      p=array / array.sum())

        array = array[::-1]
        loser_idx = np.random.choice(array,
                                     size=n_reproduction,
                                     replace=False,
                                     p=array / array.sum())

        winner = self.population[winner_idx]
        loser = self.population[loser_idx]

        for _ in range(self.reproduction_scale):
            # 基因交換
            children = self.geneExchange(loser=loser, winner=winner)

            children = self.mutate(children=children)

            # 加入族群中
            self.addOffspring(offspring=children)
            self.updatePopulationSize(n_population=len(self.population))
            # self.logger.info(f"#population: {self.n_population}")

    def geneExchange(self, *args, **kwargs):
        children = kwargs["loser"].copy()
        n_children, n_dna = children.shape

        # mu 和相對應的 std 一起進行交換
        half_flop = np.random.rand(n_children, int(n_dna / 2)) < self.exchange_rate
        flop = np.hstack((half_flop, half_flop))

        children[flop] = kwargs["winner"][flop]
        return children

    def mutate(self, *args, **kwargs):
        mu = self.getMu(population=kwargs["children"])
        std = self.getStd(population=kwargs["children"])
        # self.logger.debug(f"mu.shape: {mu.shape}, std.shape: {std.shape}")

        # 以原始 mu 為平均數，標準差為 std 的常態分配重新抽樣
        mu = np.random.normal(loc=mu, scale=std)

        #
        std *= (np.random.rand(*std.shape) * self.mutation_strength + 1e-5)

        return np.hstack((mu, std))

    def modifyEarlyStop(self, *args, **kwargs):
        average_fitness = kwargs['fitness'].mean()

        if average_fitness > self.fitness:
            self.fitness = average_fitness

            self.mutation_strength /= 0.8
        else:
            if self.mutation_strength < 1e-5:
                self.potential = 0

            self.mutation_strength *= 0.8

    def naturalSelection(self, *args, **kwargs):
        self.population = self.population[-self.N_POPULATION:]

    def getMu(self, population=None):
        if population is None:
            population = self.population

        return population[:, :self.rna_size]

    def getStd(self, population=None):
        if population is None:
            population = self.population

        return population[:, self.rna_size:]

    def getValidStdIndex(self):
        """
        取得有效標準差的索引值

        :return:
        """
        std = self.getStd()

        # 標準差為負數處標記為 1，RNA 其中只要至少一處為 1，加總便會大於等於 1
        invalid_std = np.array(std < 0.0).astype(np.int).sum(axis=1)

        # 有效的標準差所有標記都應為 0
        valid_idx = np.where(invalid_std <= 0.0)

        return valid_idx


class MutationStrengthDistributionDemo(FragmentEvolution):
    """ES(1+1).py"""

    def __init__(self, value_range, fragment_size, n_population, exchange_rate):
        self.value_range = value_range
        self.exchange_rate = exchange_rate
        self.mutation_strength = 2.0
        super().__init__(fragment_size=fragment_size, n_population=n_population,
                         logger_name="MutationStrengthDistributionDemo")

    def initPopulation(self):
        value_range = self.value_range[1] - self.value_range[0]

        mu = np.random.rand(self.n_population, self.rna_size) * value_range / self.rna_size
        std = np.random.rand(self.n_population, self.rna_size) * self.mutation_strength + 1e-5
        self.population = np.hstack((mu, std))

    def translation(self):
        mu = self.getMu()

        return mu.sum(axis=1)

    def evolve(self, *args, **kwargs):
        # 計算適應度並排序
        self.getFitness()

        # 繁殖
        self.reproduction()

        # 計算適應度並排序
        fitness = self.getFitness()

        # Early stop
        self.modifyEarlyStop(fitness=fitness)

        # 淘汰機制
        self.naturalSelection()

    def getFitness(self, *args, **kwargs):
        mu = self.translation()
        valid_mu = np.where((self.value_range[0] <= mu) & (mu <= self.value_range[1]))
        # self.logger.info(f"#valid_mu: {len(valid_mu[0])}")
        # valid_std = self.getValidStdIndex()
        # self.logger.info(f"#valid_std: {len(valid_std[0])}")
        # valid_idx = np.intersect1d(valid_mu, valid_std)
        valid_idx = valid_mu

        # 淘汰超出值域範圍的基因組
        self.population = self.population[valid_idx]
        self.updatePopulationSize(n_population=len(self.population))
        # self.logger.info(f"#population: {self.n_population}")

        # 計算適應度
        fitness = func(self.translation())

        # 根據 values 數值大小，給予排名的數列，數值越小，排名數值越小
        # np.argsort([9, 4, 6]) -> array([1, 2, 0], dtype=int64)
        indexs = np.argsort(fitness)

        # 根據適應度排序: 讓最小的排最前面，最大的排最後面
        self.population = self.population[indexs]

        return fitness

    def reproduction(self, *args, **kwargs):
        n_reproduction = int(self.n_population * self.reproduction_rate)

        # 根據適應度換算成機率，越高被選擇到的機率則越高，適應度低的基因組也有機會被選到，但機率也比較低
        array = np.arange(self.n_population)
        winner_idx = np.random.choice(array,
                                      size=n_reproduction,
                                      replace=False,
                                      p=array / array.sum())

        array = array[::-1]
        loser_idx = np.random.choice(array,
                                     size=n_reproduction,
                                     replace=False,
                                     p=array / array.sum())

        winner = self.population[winner_idx]
        loser = self.population[loser_idx]

        for _ in range(self.reproduction_scale):
            # 基因交換
            children = self.geneExchange(loser=loser, winner=winner)

            children = self.mutate(children=children)

            # 加入族群中
            self.addOffspring(offspring=children)
            self.updatePopulationSize(n_population=len(self.population))
            # self.logger.info(f"#population: {self.n_population}")

    def geneExchange(self, *args, **kwargs):
        children = kwargs["loser"].copy()
        # self.logger.debug(f"children.shape: {children.shape}")
        n_children, n_dna = children.shape

        # mu 和相對應的 std 一起進行交換
        half_flop = np.random.rand(n_children, int(n_dna / 2)) < self.exchange_rate
        flop = np.hstack((half_flop, half_flop))

        children[flop] = kwargs["winner"][flop]
        return children

    def mutate(self, *args, **kwargs):
        mu = self.getMu(population=kwargs["children"])
        std = self.getStd(population=kwargs["children"])
        # self.logger.debug(f"mu.shape: {mu.shape}, std.shape: {std.shape}")

        # 以原始 mu 為平均數，標準差為 std 的常態分配重新抽樣
        mu = np.random.normal(loc=mu, scale=std)

        # 以原始 std 為平均數，標準差為 1 的常態分配重新抽樣
        std *= (np.random.rand(*std.shape) * self.mutation_strength + 1e-5)

        return np.hstack((mu, std))

    def modifyEarlyStop(self, *args, **kwargs):
        # ES(1 + 1)
        p_target = 1 / 5

        average_fitness = kwargs['fitness'].mean()

        # 子代比親代優秀 -> MUT_STRENGTH *= 大 -> 持續變異，探索可能空間 2.028114981647472
        if average_fitness > self.fitness:
            self.fitness = average_fitness

            ps = 1.0

        # 親代比子代優秀 -> MUT_STRENGTH *= 小 -> 變異收斂 0.8379668855787558
        else:
            ps = 0.0

        self.mutation_strength *= np.exp(1 / np.sqrt(self.rna_size + 1) * (ps - p_target) / (1 - p_target))

        if self.mutation_strength < 1e-5:
            self.potential = 0

    def naturalSelection(self, *args, **kwargs):
        self.population = self.population[-self.N_POPULATION:]

    def getMu(self, population=None):
        if population is None:
            population = self.population

        return population[:, :self.rna_size]

    def getStd(self, population=None):
        if population is None:
            population = self.population

        return population[:, self.rna_size:]

    def getValidStdIndex(self):
        """
        取得有效標準差的索引值

        :return:
        """
        std = self.getStd()

        # 標準差為負數處標記為 1，RNA 其中只要至少一處為 1，加總便會大於等於 1
        invalid_std = np.array(std < 0.0).astype(np.int).sum(axis=1)

        # 有效的標準差所有標記都應為 0
        valid_idx = np.where(invalid_std <= 0.0)

        return valid_idx


if __name__ == "__main__":
    FRAGMENT_SIZE = 10
    N_POPULATION = 200
    EXCHANGE_RATE = 0.6
    MUTATION_RATE = 0.003
    X_BOUND = [0, 5]
    N_GENERATIONS = 200


    def testSimpleExchangeDemo():
        """
        將畫圖相關程式碼移到函式中，會導致互動效果出錯，但並非程式碼本身出錯。

        :return:
        """
        sed = SimpleExchangeDemo(value_range=X_BOUND,
                                 fragment_size=FRAGMENT_SIZE,
                                 n_population=N_POPULATION,
                                 mutation_rate=MUTATION_RATE)
        sed.setPotential(potential=5)

        plt.ion()
        x = np.linspace(*X_BOUND, 200)
        plt.plot(x, func(x))
        sca = None

        for gen in range(N_GENERATIONS):
            if sca is not None:
                sca.remove()

            x = sed.translation()
            y = func(x)
            # print(y)
            sca = plt.scatter(x,
                              y,
                              s=200,
                              lw=0,
                              c='red',
                              alpha=0.5)
            plt.pause(0.05)

            sed.evolve()
            print(gen, func(sed.translation()[-1]))

            if sed.potential <= 0:
                break

        plt.ioff()
        plt.show()


    def testWinnerLoserDemo():
        wld = WinnerLoserDemo(value_range=X_BOUND,
                              fragment_size=FRAGMENT_SIZE,
                              n_population=N_POPULATION,
                              exchange_rate=EXCHANGE_RATE,
                              mutation_rate=MUTATION_RATE)

        plt.ion()
        x = np.linspace(*X_BOUND, 200)
        plt.plot(x, func(x))
        sca = None

        for gen in range(N_GENERATIONS):
            if sca is not None:
                sca.remove()

            x = wld.translation()
            y = func(x)
            # print(y)
            sca = plt.scatter(x,
                              y,
                              s=200,
                              lw=0,
                              c='red',
                              alpha=0.5)
            plt.pause(0.05)

            wld.evolve()
            print(f"gen: {gen}, avg: {wld.fitness}, best: {func(wld.translation()[-1])}")

            if wld.potential <= 0:
                break

        plt.ioff()
        plt.show()


    def testDistributionDemo():
        dd = DistributionDemo(value_range=X_BOUND,
                              fragment_size=FRAGMENT_SIZE,
                              n_population=N_POPULATION,
                              exchange_rate=EXCHANGE_RATE)
        dd.setPotential(potential=20)

        plt.ion()
        x = np.linspace(*X_BOUND, 200)
        plt.plot(x, func(x))
        sca = None

        for gen in range(N_GENERATIONS):
            if sca is not None:
                sca.remove()

            x = dd.translation()
            y = func(x)
            # print(y)
            sca = plt.scatter(x,
                              y,
                              s=200,
                              lw=0,
                              c='red',
                              alpha=0.5)
            plt.pause(0.05)

            dd.evolve()
            print(f"gen: {gen}, avg: {dd.fitness}, best: {func(dd.translation()[-1])}, potential: {dd.potential}")

            if dd.potential <= 0:
                break

        plt.ioff()
        plt.show()


    def testMutationStrengthDistributionDemo():
        msdd = MutationStrengthDistributionDemo(value_range=X_BOUND,
                                                fragment_size=FRAGMENT_SIZE,
                                                n_population=N_POPULATION,
                                                exchange_rate=EXCHANGE_RATE)
        msdd.setPotential(potential=20)

        plt.ion()
        x = np.linspace(*X_BOUND, 200)
        plt.plot(x, func(x))
        sca = None

        for gen in range(N_GENERATIONS):
            if sca is not None:
                sca.remove()

            x = msdd.translation()
            y = func(x)
            # print(y)
            sca = plt.scatter(x,
                              y,
                              s=200,
                              lw=0,
                              c='red',
                              alpha=0.5)
            plt.pause(0.05)

            msdd.evolve()
            print(f"gen: {gen}, avg: {msdd.fitness}, "
                  f"best: {func(msdd.translation()[-1])}, mutation_strength: {msdd.mutation_strength}")

            if msdd.potential <= 0:
                break

        plt.ioff()
        plt.show()
