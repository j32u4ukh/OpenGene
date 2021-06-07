import numpy as np
from numpy import nonzero


class Gene:
    # TODO: 產生基因後，便可根據結構以及各個細胞所需基因個數，檢查該基因序列是否能夠建構出生命體
    # 利用 classmethod 的形式，避免 normalizer 的重複計算，減少計算量
    sign_normalizer = None
    unsign_normalizer = None

    # ===== 定義其他共用參數 =====
    # 結構基因組長度：以多少的基因來定義一個結構參數
    struct_digits = 2

    # 數值基因組長度：以多少的基因來定義一個最小數值
    value_digits = 8

    # 間隔幾個基因再取下一個基因組
    value_steps = 4

    @classmethod
    def signValue(cls, genome: np.array):
        if cls.sign_normalizer is None:
            cls.sign_normalizer = translate([1.0 for _ in range(len(genome) - 1)], start=0, base=2.0)

        translated = translate(genome[1:], start=0, base=2.0)

        if genome[0] == 1:
            translated *= -1.0

        translated /= cls.sign_normalizer

        return translated

    @classmethod
    def unsignValue(cls, genome: np.array):
        if cls.unsign_normalizer is None:
            cls.unsign_normalizer = translate([1.0 for _ in range(len(genome))], start=0, base=2.0)

        translated = translate(genome, start=0, base=2.0)
        translated /= cls.unsign_normalizer

        return translated


# 產生 n_gene 個 0/1 基因
def createGene(n_gene):
    gene = np.random.randint(low=0, high=2, size=n_gene)
    return gene


def translate(genome, start: int = 0, base: float = 1.0):
    n_gene = len(genome)
    start -= 1

    # 指數次方
    exponent = np.arange(start + n_gene, start, -1)

    if base == 1.0:
        # 1 的任何次方都是 1，因此 base = 1 為例外 -> ..., 3, 2, ..., start
        multiplier = exponent
    else:
        # multiplier = base^exponent ex: 2^3 = 8
        multiplier = base ** exponent

    return (genome * multiplier).sum()


# 轉譯架構類基因(不同於定義數值的基因)
def translateStruct(genome: np.array):
    translated = translate(genome, start=0, base=2.0)

    return int(translated + 1)


# 計算漢明距離
def hamming(a: np.array, b: np.array, normalize=False):
    hamm = len(nonzero(a != b)[0])

    if normalize:
        hamm /= len(a)

    return hamm


if __name__ == "__main__":
    genome = createGene(n_gene=8)
    print("genome:", genome)

    struct_index = translateStruct(genome)
    print("struct_index:", struct_index)

    unsign_value = Gene.unsignValue(genome)
    print("unsign_value:", unsign_value)

    sign_value = Gene.signValue(genome)
    print("sign_value:", sign_value)
