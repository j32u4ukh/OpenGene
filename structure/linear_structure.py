import numpy as np

from structure import Structure


# 直線型串接的基因組
class LinearStructure(Structure):
    def __init__(self, gene: np.array, n_node: int):
        super().__init__()
        self.list_node = self.buildStructure(gene=gene, n_node=n_node)

    def buildStructure(self, gene: np.array, n_node: int):
        """
        將一維陣列的 gene 轉化為二維陣列的 matrix，在索引值的讀取上有較好的表現。
        在發生 IndexError 的時候，可以確保索引值會正確增加與取得。

        :param gene: 定義結構的基因(一維陣列 shape = (n_node*n_node,))
        :param n_node: 細胞個數
        :return:
        """
        matrix = gene.reshape((n_node, n_node))
        nodes = []

        # 根據結構矩陣，產生 ListNode
        for i in range(n_node):
            try:
                j = np.where(matrix[i] == 1.0)[0][0]
            except IndexError:
                # LinearStructure 是一個接一個，因此 5 個細胞只會有 4 個連結，因而產生 IndexError
                continue

            # 產生節點(沒有實作的細胞稱為節點)
            list_node = ListNode(i)
            list_node.add(j)

            # 利用 cells 管理所有節點
            nodes.append(list_node)

        # 檢查 ListNode 的頭尾連結關係
        node1, node2 = ListNode.checkLinkable(nodes)
        is_linkable = (node1 is not None) and (node2 is not None)

        while is_linkable:
            # 將 nodes[ln2] 加到 nodes[ln1] 之後，並將 nodes[ln2] 移除
            nodes[node1] += nodes[node2]
            del nodes[node2]

            # 檢查 ListNode 的頭尾連結關係
            node1, node2 = ListNode.checkLinkable(nodes)
            is_linkable = (node1 is not None) and (node2 is not None)

        # 由於為"直線型串接"，理論上只會有一個 ListNode 才對
        return nodes[0]

    def loadCells(self, cells):
        self.list_node.setCell(cell=cells[self.list_node.node_id])
        curr_node = self.list_node.next_node

        while curr_node is not None:
            curr_node.setCell(cell=cells[curr_node.node_id])
            curr_node = curr_node.next_node


class ListNode:
    def __init__(self, node_id, root=None):
        self.root = root
        self.node_id = node_id
        self.cell = None

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
            self.next_node = ListNode(node_id=next_id, root=self)
        else:
            self.next_node.add(next_id=next_id)

    def lastNode(self):
        if self.next_node is not None:
            return self.next_node.lastNode()
        else:
            return self

    def lastNodeId(self):
        return self.lastNode().node_id

    def setCell(self, cell):
        # 若是網狀結構，則需檢查當前 cell 是否為 None，甚至'下一個 cell'是否為 None，以避免環狀結構
        self.cell = cell


# 產生"線性的"結構定義矩陣(只會 a -> b 不會 a -> b, c 或是 a -> b & b -> a 形成循環)
def createLinearStructure(n_cell):
    matrix = np.zeros((n_cell, n_cell))
    sequence = np.arange(0, n_cell)
    np.random.shuffle(sequence)
    print(f"sequence: {sequence}")

    x = sequence[0]

    for i in range(1, n_cell):
        y = sequence[i]

        matrix[x, y] = 1.0
        x = y

    structure_gene = matrix.reshape(-1)
    return structure_gene


if __name__ == "__main__":
    def testCreater(n_cell=5):
        structure_gene = createLinearStructure(n_cell=n_cell)
        print(structure_gene)

        matrix = structure_gene.reshape((n_cell, n_cell))
        print(matrix)


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

    # 分析結構基因，構成 ListNode
    def testParseStructure(n_node):
        gene = createLinearStructure(n_node)
        matrix = gene.reshape((n_node, n_node))
        print(f"matrix:\n{matrix}")
        nodes = []

        # 根據結構矩陣，產生 ListNode
        for i in range(n_node):
            try:
                j = np.where(matrix[i] == 1.0)[0][0]
            except IndexError:
                continue

            # 產生節點(沒有實作的細胞稱為節點)
            list_node = ListNode(i)
            list_node.add(j)
            print(f"list_node: {list_node}")

            # 利用 cells 管理所有節點
            nodes.append(list_node)

        # 檢查 ListNode 的頭尾連結關係
        node1, node2 = ListNode.checkLinkable(nodes)
        is_linkable = (node1 is not None) and (node2 is not None)

        while is_linkable:
            # 將 nodes[ln2] 加到 nodes[ln1] 之後，並將 nodes[ln2] 移除
            nodes[node1] += nodes[node2]
            del nodes[node2]

            # 檢查 ListNode 的頭尾連結關係
            node1, node2 = ListNode.checkLinkable(nodes)
            is_linkable = (node1 is not None) and (node2 is not None)

        # 由於為"直線型串接"，理論上只會有一個 ListNode 才對
        print(nodes[0])

    def testLinearStructure(n_node=8):
        gene = createLinearStructure(n_node)
        linear_structure = LinearStructure(gene=gene, n_node=n_node)


    # testCreater()
    # testListNode()
    testParseStructure(n_node=8)
