import numpy as np

from evolution import Evolution


# TODO: strategy.getRevenu() / getIncome
def getRevenus(strategys):
    revenus = []

    for strategy in strategys:
        revenu = strategy.getRevenu()
        revenus.append(revenu)

    return np.array(revenus)


def getTradeTimes(strategys):
    trade_times = []

    for strategy in strategys:
        trade_time = strategy.getTradeTime()
        trade_times.append(trade_time)

    return np.array(trade_times)


# TODO: 基因組的讀寫 for 基因組擴展性
class StrategyEvolution(Evolution):
    def __init__(self, n_population,
                 stock_id, start_date, end_date=None, datetime_index=False, h_start_date=None, h_end_date=None):
        super().__init__(n_population=n_population, logger_name="StrategyEvolution")
        self.stock_id = stock_id
        self.start_date = start_date
        self.end_date = end_date
        self.datetime_index = datetime_index
        self.h_start_date = h_start_date
        self.h_end_date = h_end_date

    # 族群初始化
    def initPopulation(self):
        # TODO: 0100110101010101
        pass

    # TODO: 多組策略 -> 一組策略中有多個基因組
    def translation(self):
        """
        將基因組轉換成策略

        :return: 多組策略
        """
        strategys = []

        # TODO: EvolutionStrategy 為策略，透過 StrategyEvolution 來初始化，本身不需要演化或繁殖，只需最後返回收益表現，
        #  用以讓 StrategyEvolution 判斷誰可以存續，誰又該淘汰
        # for i in range(self.n_population):
        #     strategy = EvolutionStrategy(dna=self.population[i],
        #                                  fragment_size=self.fragment_size,
        #                                  n_fragment=self.n_fragment)
        #
        #     strategys.append(strategy)

        return strategys

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
        # 轉譯成多組策略
        strategys = self.translation()

        # 回測載入多組策略
        # TODO: Backtest
        # backtest = Backtest(strategys=strategys)
        # backtest.run(self.stock_id,
        #              start_date=self.start_date,
        #              end_date=self.end_date,
        #              datetime_index=self.datetime_index,
        #              h_start_date=self.h_start_date,
        #              h_end_date=self.h_end_date)

        # 計算適應度(取得各策略的最終收益 -> 增加額外計算要素如:交易次數)
        revenus = getRevenus(strategys=strategys)
        n_trades = getTradeTimes(strategys=strategys)
        fitness = self.computeFitness(revenus, n_trades)

        # 根據 values 數值大小，給予排名的數列，數值越小，排名數值越小
        # np.argsort([9, 4, 6]) -> array([1, 2, 0], dtype=int64)
        index = np.argsort(fitness)

        # 根據適應度排序: 讓最小的排最前面，最大的排最後面
        self.population = self.population[index]

        return fitness

    def reproduction(self, *args, **kwargs):
        pass

    def geneExchange(self, *args, **kwargs):
        pass

    def mutate(self, *args, **kwargs):
        pass

    def modifyEarlyStop(self, *args, **kwargs):
        average_fitness = kwargs['fitness'].mean()

        if average_fitness > self.fitness:
            self.fitness = average_fitness

            self.mutation_strength /= self.mutation_scale
        else:
            if self.mutation_strength < self.early_stop:
                self.potential = 0

            self.mutation_strength *= self.mutation_scale

    def naturalSelection(self, *args, **kwargs):
        # 只保留適應度大於 0 的基因組
        self.population = self.population[np.where(kwargs['fitness'] > 0.0)]
        self.logger.info(f"#(fitness > 0.0): {len(self.population)}")
        self.population = self.population[-self.N_POPULATION:]
        self.updatePopulationSize(n_population=len(self.population))

    """ For strategy """
    @staticmethod
    def computeFitness(revenus, n_trades):
        return revenus / n_trades
