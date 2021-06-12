import numpy as np

from structure import Structure


# 直線型串接的基因組
class LinearStructure(Structure):
    def __init__(self):
        super().__init__()

    def parseStructure(self, matrix):
        cells = []
        rows = len(matrix)

        # 根據結構矩陣，產生 ListCell
        for row in range(rows):
            try:
                idx = np.where(matrix[row] == 1.0)[0][0]
            except IndexError:
                continue

            list_cell = ListNode(row)
            list_cell.add(idx)

            cells.append(list_cell)

        # 檢查 ListNode 的頭尾連結關係
        ln1, ln2 = ListNode.checkLinkable(cells)
        is_linkable = (ln1 is not None) and (ln2 is not None)

        while is_linkable:
            # 將 nodes[ln2] 加到 nodes[ln1] 之後，並將 nodes[ln2] 移除
            cells[ln1] += cells[ln2]
            del cells[ln2]

            # 檢查 ListNode 的頭尾連結關係
            ln1, ln2 = ListNode.checkLinkable(cells)
            is_linkable = (ln1 is not None) and (ln2 is not None)

        # 由於為"直線型串接"，理論上只會有一個 ListNode 才對
        return cells[0]


class ListNode:
    def __init__(self, node_id, root=None):
        self.root = root
        self.node_id = node_id
        self.next_node = None

    def __add__(self, other):
        self.lastNode().root.next_node = other

        return self

    def __str__(self):
        info = f"ListNode({self.node_id}"
        next_node = self.next_node

        while next_node is not None:
            info += f" -> {next_node.node_id}"
            next_node = next_node.next_node

        info += ")"

        return info

    __repr__ = __str__

    # 檢查多個 ListNode 之間是否有頭尾連結關係
    @staticmethod
    def checkLinkable(nodes):
        n_node = len(nodes)

        for i in range(n_node):
            for j in range(n_node):
                if i == j:
                    continue

                if nodes[i].lastNodeId() == nodes[j].node_id:
                    return i, j

        return None, None

    def add(self, next_id):
        if self.next_node is None:
            self.next_node = ListNode(next_id, root=self)
        else:
            self.next_node.add(next_id)

    def lastNode(self):
        if self.next_node is not None:
            return self.next_node.lastNode()
        else:
            return self

    def lastNodeId(self):
        return self.lastNode().node_id


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
        print("tail1:", list_node1.lastNodeId())

        list_node2 = ListNode(3)
        list_node2.add(4)
        list_node2.add(5)
        print(list_node2)
        print("tail2:", list_node2.lastNodeId())

        list_node3 = ListNode(4)
        list_node3.add(5)
        list_node3.add(6)
        print(list_node3)
        print("tail3:", list_node3.lastNodeId())

        nodes = [list_node2, list_node1, list_node3]
        lc1, lc2 = ListNode.checkLinkable(nodes)
        print(f"lc1: {lc1}, lc2: {lc2}")

        is_linkable = (lc1 is not None) and (lc2 is not None)

        while is_linkable:
            nodes[lc1] += nodes[lc2]
            del nodes[lc2]

            lc1, lc2 = ListNode.checkLinkable(nodes)
            print(f"lc1: {lc1}, lc2: {lc2}")

            is_linkable = (lc1 is not None) and (lc2 is not None)

        for node in nodes:
            print(node)

    def testParseStructure():
        from sys import getsizeof

        cells = []
        matrix = createLinearStructure(n_cell=10)
        print(matrix)
        print(f"size of matrix:", getsizeof(matrix))

        rows = len(matrix)
        for row in range(rows):
            try:
                idx = np.where(matrix[row] == 1.0)[0][0]
            except IndexError:
                continue

            list_cell = ListNode(row)
            list_cell.add(idx)

            cells.append(list_cell)

        for i, cell in enumerate(cells):
            print(i, cell)

        print("==================================================")

        lc1, lc2 = ListNode.checkLinkable(cells)

        is_linkable = (lc1 is not None) and (lc2 is not None)

        while is_linkable:
            cells[lc1] += cells[lc2]
            del cells[lc2]

            lc1, lc2 = ListNode.checkLinkable(cells)
            is_linkable = (lc1 is not None) and (lc2 is not None)

        for i, cell in enumerate(cells):
            print(i, cell)

        print(f"size of ListCell:", getsizeof(cells[0]))


    # testListNode()
    testParseStructure()
