import numpy as np


class GeneNode:
    def __init__(self, val):
        self.val = val
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

    def __add__(self, other):
        """
        從 root 連接到 leaf

        :param other: 其他 GeneNode
        :return:
        """
        # root: 上層 GeneNode 數值
        # leaf: 下層 GeneNode 數值
        if self.val == other.root:
            for node in self.nodes:
                if node.val == other.leaf:
                    print(f"已有 GeneNode({other.leaf})")
                    break

            self.nodes.append(GeneNode(val=other.leaf))
        else:
            for node in self.nodes:
                if node.val == other.root:
                    node.add(GeneNode(val=other.leaf))

    def __sub__(self, other):
        if self.val == other.root:
            n_node = len(self.nodes)

            for i in range(n_node):
                node = self.nodes[i]

                if node.val == other.val:
                    self.nodes = [self.nodes[j] for j in range(n_node) if j != i]
                    break

        for node in self.nodes:
            if node.val == other.root:
                node.__sub__(other)
                break

    def add(self, root, leaf):
        """

        :param root: 上層 GeneNode 數值
        :param leaf: 下層 GeneNode 數值
        :return:
        """
        if self.val == root:
            for node in self.nodes:
                if node.val == leaf:
                    print(f"已有 GeneNode({leaf})")
                    break

            self.nodes.append(GeneNode(val=leaf))
        else:
            for node in self.nodes:
                if node.val == root:
                    node.add(GeneNode(val=leaf))

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


def createGenome(length):
    gene = np.random.randint(low=0, high=2, size=length)
    str_gene = str(gene).strip("[]").replace(" ", "")
    return str_gene


def construct(structure_genome):
    # TODO: 排除自己連接到自己
    xs, ys = np.where(structure_genome == 1)
    x_set, y_set = set(), set()
    order_dict = dict()

    for (x, y) in zip(xs, ys):
        if not order_dict.__contains__(x):
            order_dict[x] = []

        if x != y:
            order_dict[x].append(y)

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

