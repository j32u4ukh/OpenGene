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


class Vertex:
    def __init__(self, vertex_id):
        self.vertex_id = vertex_id
        self.forward = []
        self.backward = []

    def __str__(self):
        forward_vertex = [vertex.vertex_id for vertex in self.forward]
        info = f"Vertex({self.vertex_id} -> {forward_vertex})"

        return info

    __repr__ = __str__

    def addForward(self, other):
        self.forward.append(other)

    def addBackward(self, other):
        self.backward.append(other)

    def getDepth(self, reverse: bool = False):
        if reverse:
            ward = self.backward
        else:
            ward = self.backward

        if len(ward) == 0:
            return 0
        else:
            depth = 0

            for w in ward:
                depth = max(depth, w.getDepth())

            return depth

    # 將 forward 以及 forward 的 forward 當中的'自己'移除，避免環狀結構
    def prune(self):
        pass


class Graph:
    def __init__(self):
        self.vertices = []

    def __str__(self):
        info = "Graph"

        for vertex in self.vertices:
            info += f"\n{vertex}"

        return info

    __repr__ = __str__

    @staticmethod
    def addEdge(vertex1: Vertex, vertex2: Vertex):
        vertex1.addForward(vertex2)
        vertex2.addBackward(vertex1)

    def addEdgeById(self, id1: int, id2: int):
        vertex1 = self.findVertex(vertex_id=id1)
        vertex2 = self.findVertex(vertex_id=id2)

        if vertex1 is not None and vertex2 is not None:
            print(f"({id1}).addForward({id2})")
            vertex1.addForward(vertex2)
            print(f"({id2}).addBackward({id1})")
            vertex2.addBackward(vertex1)
            print()

    def loadAdjacencyMatrix(self, matrix):
        pass

    def addVertex(self, vertex: Vertex):
        if not self.isContainVertex(vertex_id=vertex.vertex_id):
            self.vertices.append(vertex)

    def addVertexById(self, vertex_id: int):
        if not self.isContainVertex(vertex_id=vertex_id):
            self.vertices.append(Vertex(vertex_id=vertex_id))

    def isContainVertex(self, vertex_id: int):
        for vertex in self.vertices:
            if vertex.vertex_id == vertex_id:
                return True

        return False

    def findVertex(self, vertex_id: int):
        for vertex in self.vertices:
            if vertex.vertex_id == vertex_id:
                return vertex

        return None

    # 將 forward 以及 forward 的 forward 當中的親頂點移除
    def prune(self):
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
    print(gene)

    nodes = []


    def findConnection(nodes):
        n_node = len(nodes)
        idx = 0
        connection = []

        for i in range(n_node):
            # TODO: 不是找最後一個節點，而是找路徑上是否有可以連結兩棵 GraphTree 的地方
            leaves_id = nodes[i].getLeavesId()
            print(f"LeavesId of nodes[{i}]: {leaves_id}")

            for j in range(n_node):
                if i == j:
                    continue

                node_id = nodes[j].node_id
                print(f"j nodes[{j}]: {node_id}")
                if node_id in leaves_id:
                    connection.append(j)

            if len(connection) > 0:
                idx = i
                break

        return idx, connection


    graph = Graph()

    # 初始化
    for i in range(n_cell):
        structure_gene = gene[i]
        connection_indexs = np.where(structure_gene == 1.0)[0]

        if len(connection_indexs) > 0:
            graph.addVertexById(vertex_id=i)

            for index in connection_indexs:
                graph.addVertexById(vertex_id=index)
                graph.addEdgeById(id1=i, id2=index)

    for vertex in graph.vertices:
        depth = vertex.getDepth()
        print(depth, vertex)
