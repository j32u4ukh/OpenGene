import numpy as np

from cell import Cell
from structure import Structure


# 直線型串接的基因組
class LinearStructure(Structure):
    def __init__(self):
        super().__init__(cells=[])

    def parseStructure(self, matrix):
        pass

    def add(self, cell: Cell):
        self.cells.append(cell)

    def run(self, x):
        for cell in self.cells:
            x = cell.call(x)

        return x


class ListNode:
    def __init__(self, val, root=None):
        self.root = root
        self.val = val
        self.next_node = None

    def __add__(self, other):
        self.lastNode().root.next_node = other

        return self

    def __str__(self):
        info = f"ListNode({self.val}"
        next_node = self.next_node

        while next_node is not None:
            info += f" -> {next_node.val}"
            next_node = next_node.next_node

        info += ")"

        return info

    __repr__ = __str__

    @staticmethod
    def checkLinkable(nodes):
        n_node = len(nodes)

        for i in range(n_node):
            for j in range(n_node):
                if i == j:
                    continue

                if nodes[i].lastValue() == nodes[j].val:
                    return i, j

        return None, None

    def add(self, next_val):
        if self.next_node is None:
            self.next_node = ListNode(next_val, root=self)
        else:
            self.next_node.add(next_val)

    def lastNode(self):
        if self.next_node is not None:
            return self.next_node.lastNode()
        else:
            return self

    def lastValue(self):
        return self.lastNode().val


# 產生"線性的"結構定義矩陣
def createLinearStructure(n_cell):
    matrix = np.zeros((n_cell, n_cell))
    sequence = np.arange(0, n_cell)
    np.random.shuffle(sequence)
    print(f"sequence: {sequence}")

    x = sequence[0]

    for i in range(1, n_cell):
        y = sequence[i]

        matrix[x, y] = 1
        x = y

    return matrix


if __name__ == "__main__":
    def testListNode():
        list_node1 = ListNode(1)
        list_node1.add(2)
        list_node1.add(3)
        print(list_node1)
        print("tail1:", list_node1.lastValue())

        list_node2 = ListNode(3)
        list_node2.add(4)
        list_node2.add(5)
        print(list_node2)
        print("tail2:", list_node2.lastValue())

        list_node3 = ListNode(4)
        list_node3.add(5)
        list_node3.add(6)
        print(list_node3)
        print("tail3:", list_node3.lastValue())

        nodes = [list_node2, list_node1, list_node3]
        ln1, ln2 = ListNode.checkLinkable(nodes)
        print(f"ln1: {ln1}, ln2: {ln2}")

        is_linkable = (ln1 is not None) and (ln2 is not None)

        while is_linkable:
            nodes[ln1] += nodes[ln2]
            del nodes[ln2]

            ln1, ln2 = ListNode.checkLinkable(nodes)
            print(f"ln1: {ln1}, ln2: {ln2}")

            is_linkable = (ln1 is not None) and (ln2 is not None)

        for node in nodes:
            print(node)

    # ls = LinearStructure()

    # parseStructure
    nodes = []
    matrix = createLinearStructure(n_cell=10)
    print(matrix)

    rows = len(matrix)
    for row in range(rows):
        try:
            idx = np.where(matrix[row] == 1.0)[0][0]
        except IndexError:
            continue

        list_node = ListNode(row)
        list_node.add(idx)

        nodes.append(list_node)

    for i, node in enumerate(nodes):
        print(i)
        print(node)

    print("==================================================")

    ln1, ln2 = ListNode.checkLinkable(nodes)

    is_linkable = (ln1 is not None) and (ln2 is not None)

    while is_linkable:
        nodes[ln1] += nodes[ln2]
        del nodes[ln2]

        ln1, ln2 = ListNode.checkLinkable(nodes)
        print(f"ln1: {ln1}, ln2: {ln2}")

        is_linkable = (ln1 is not None) and (ln2 is not None)

    for i, node in enumerate(nodes):
        print(i)
        print(node)
