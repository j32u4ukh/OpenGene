import numpy as np


# TODO: 或許應直接發展網狀結構，但輸入直線型的基因定義，直接退化成直線型
# TODO: 網狀結構類別，用於管理細胞結構的管理
# TODO: 不允許環狀結構，即根節點不能同時作為子節點或孫節點
class Structure:
    def __init__(self):
        pass

    # 根據節構基因，形成 Structure
    def buildStructure(self, gene: np.array, n_cell: int):
        pass

    def loadCells(self, cells):
        pass


# TODO: 建構重點在於避免將根節點作為子節點加入，子節點本身可能沒問題，但孫節點可能有問題，需再進一步做修剪
class GraphNode:
    def __init__(self, node_id, roots: list = None, leaves: list = None):
        self.node_id = node_id
        self.layer = 0

        if roots is None:
            self.roots = []
        else:
            self.roots = roots

        if leaves is None:
            self.leaves = []
        else:
            self.leaves = leaves

    def __str__(self):
        leaves = [leaf.node_id for leaf in self.leaves]
        info = f"GraphNode({self.node_id} -> {leaves})"

        return info

    def __add__(self, other):
        last_leaf = self.getLastLeaf(node_id=other.node_id)

        if last_leaf is not None:
            last_leaf.addLeaf(*other.leaves)

        return self

    def addRoot(self, *roots):
        for root in roots:
            self.roots.append(root)

    def addLeaf(self, *leaves):
        for leaf in leaves:
            self.leaves.append(leaf)

    def setLayer(self, layer):
        self.layer = layer

    def getLastLeaves(self):
        leaves = []

        for leaf in self.leaves:
            print(f"getLastLeaves | {leaf.node_id}")
            sub_leaves = leaf.leaves

            if len(sub_leaves) == 0:
                leaves.append(leaf)
            else:
                for sub_leaf in sub_leaves:
                    sub_last_leaves = sub_leaf.getLastLeaves()
                    leaves += sub_last_leaves

        return leaves

    def getLastLeaf(self, node_id):
        leaves = self.getLastLeaves()

        for leaf in leaves:
            print(f"getLastLeaf | {leaf.node_id}")
            if leaf.node_id == node_id:
                return leaf

        return None

    def getLeavesId(self):
        last_leaves = self.getLastLeaves()
        return [leaf.node_id for leaf in last_leaves]

    def isContainRoot(self, node_id):
        if len(self.roots) > 0:
            # 檢查在 roots 當中是否已有 node_id
            for root in self.roots:
                if node_id == root.node_id:
                    return True

            # 檢查在 roots 的 root 當中是否已有 node_id
            for root in self.roots:
                if root.isContainRoot(node_id):
                    return True

        return False

    # 檢查
    def pruneLeaf(self):
        pass


def createGraphStructure(n_cell, p_rate=0.1):
    gene = np.zeros((n_cell, n_cell))
    p = np.random.random((n_cell, n_cell))
    node_indexs = np.where(p < p_rate)
    gene[node_indexs] = 1.0
    np.fill_diagonal(gene, 0.0)

    return gene.flatten()


if __name__ == "__main__":
    n_cell = 10
    gene = createGraphStructure(n_cell=n_cell, p_rate=0.1)
    # print(f"gene -> {gene.shape}\n{gene}")

    gene = gene.reshape((n_cell, n_cell))
    # print(gene)

    nodes = []

    def findConnection(nodes):
        n_node = len(nodes)
        idx = 0
        connection = []

        for i in range(n_node):
            leaves_id = nodes[i].getLeavesId()

            for j in range(n_node):
                if i == j:
                    continue

                if nodes[j].node_id in leaves_id:
                    connection.append(j)

            if len(connection) > 0:
                idx = i
                break

        return idx, connection

    # 初始化
    for i in range(n_cell):
        structure_gene = gene[i]
        leaves_index = np.where(structure_gene == 1.0)[0]

        if len(leaves_index) > 0:
            leaves = [GraphNode(node_id=index) for index in leaves_index]
            node = GraphNode(node_id=i, leaves=leaves)
            print(node)
            nodes.append(node)

    idx, connection = findConnection(nodes=nodes)
    print(idx, connection)
