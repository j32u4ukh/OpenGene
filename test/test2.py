import numpy as np
from matplotlib import pyplot as plt


class GeneNode:
    def __init__(self, val, root=None):
        # 上一層 GeneNode 的 數值/id
        self.root = root

        # 這個 GeneNode 的 數值/id
        self.val = val

        # 下一層的 GeneNode 們
        self.nodes = []

    def __len__(self):
        n_node = len(self.nodes)

        if n_node == 0:
            return 1
        else:
            length = 0

            for node in self.nodes:
                length = max(length, len(node))

            return 1 + length

    def __str__(self):
        result = [self]
        length = len(result)
        idx = 0

        while idx < length:
            root = result[idx]

            for node in root.nodes:
                result.append(node)

            idx += 1
            length = len(result)

        info = f"GeneNode({self.__len__()}) | {result[0].val}"

        for i in range(1, length):
            info += f", {result[i].val}"

        return info

    __repr__ = __str__

    def add(self, other):
        """

        :param other: 其他 GeneNode
        :return:
        """
        if self.val == other.root:
            for node in self.nodes:
                if node.val == other.val:
                    print(f"已有 GeneNode({other.val})")
                    break

            self.nodes.append(GeneNode(val=other.val, root=self.val))
        else:
            for node in self.nodes:
                if node.val == other.root:
                    node.add(GeneNode(val=other.val))

    def sub(self, other):
        # 移除當前 GeneNode 連結到 other 之間的連線，other 及其子 GeneNode 則返回
        remainder = None

        if self.val == other.root:
            n_node = len(self.nodes)

            for i in range(n_node):
                node = self.nodes[i]

                if node.val == other.val:
                    self.nodes = [self.nodes[j] for j in range(n_node) if j != i]
                    remainder = node
                    break

        for node in self.nodes:
            if node.val == other.root:
                node.__sub__(other)
                break

        return remainder

    def inLeaf(self, other):
        """
        判斷 other 的 root 是否和自己的 leaf / leaf 的 leaf / ... 相同

        :param other:
        :return:
        """
        for node in self.nodes:
            if node.val == other.val or node.isLeaf(node):
                return True

        return False


def createGenome(n_gene, digits=8):
    high = pow(2, digits)
    str_format = "{0:0" + str(digits) + "b}"
    gene = np.random.randint(low=0, high=high, size=n_gene)
    str_gene = ""

    for g in gene:
        str_gene += str_format.format(g)

    return str_gene


def construct(structure_genome):
    xs, ys = np.where(structure_genome == 1)
    # x_set, y_set = set(), set()
    # order_dict = dict()
    #
    # for (x, y) in zip(xs, ys):
    #     if not order_dict.__contains__(x):
    #         order_dict[x] = []
    #
    #     if x != y:
    #         order_dict[x].append(y)
    # TODO: 排除自己連接到自己

    full = np.union1d(xs, ys)
    start = np.setdiff1d(full, ys)
    end = np.setdiff1d(full, xs)

    return start, end


def createStructureGenome(n_cell, rate=0.2):
    structure_genome = createStructureGenomeCore(n_cell=n_cell, rate=rate)
    start, end = construct(structure_genome)

    while len(start) == 0 or len(end) == 0:
        structure_genome = createStructureGenomeCore(n_cell=n_cell, rate=rate)
        start, end = construct(structure_genome)

    xs, ys = np.where(structure_genome == 1)

    return xs, ys, structure_genome


def createStructureGenomeCore(n_cell, rate=0.2):
    length = int(np.power(n_cell, 2))
    structure_genome = np.zeros((length,))
    rand = np.random.rand(length)
    indexs = np.where(rand < rate)
    structure_genome[indexs] = 1

    structure_genome = structure_genome.reshape((n_cell, n_cell))

    return structure_genome


def genomeToArray(genome):
    return np.array(list(genome), dtype=np.int)


# n_genome = 9
# genome = createGenome(n_genome)
# print(genome)
# # 01111101111001011011
#
# for i in range(n_genome - 4):
#     gene = genome[i: i + 4]
#     print(gene, int(gene, 2))

n_cell = 10
xs, ys, structure_genome = createStructureGenome(n_cell=n_cell, rate=0.2)
start, end = construct(structure_genome)
order_dict = dict()

for (x, y) in zip(xs, ys):
    if not order_dict.__contains__(x):
        order_dict[x] = []

    if x != y:
        order_dict[x].append(y)

s1 = start[0]
s2 = start[1]

n1 = GeneNode(s1)
n2 = GeneNode(s2)

for val in order_dict[s1]:
    n1.add(root=s1, leaf=val)

for val in order_dict[s2]:
    n2.add(root=s2, leaf=val)

"""
計算'前一筆數值'和'後一筆數值'之間的相關性，目前設計為'間隔基因片段長度的一半'，可以獲得低相關的數據
"""
n_gene = 12
digits = 8
gene = createGenome(n_gene=n_gene, digits=digits)
print(gene)

value = []

for i in range(0, n_gene - digits, 4):
    temp = gene[i: i + digits]
    print(i, i + digits)
    val = int(temp, 2)
    value.append(val)

n_value = len(value)

x = value[:-1]
y = value[1:]
plt.scatter(x, y)
plt.show()

print(np.corrcoef([x, y]))
