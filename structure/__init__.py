import datetime

import numpy as np

from submodule.Xu3.utils import getLogger


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
    def __init__(self, vertex_id,
                 logger_dir="graph", logger_name=datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")):
        self.logger_dir = logger_dir
        self.logger_name = logger_name
        self.extra = {"className": f"{self.__class__.__name__}_{vertex_id}"}
        self.logger = getLogger(logger_name=self.logger_name,
                                to_file=True,
                                time_file=False,
                                file_dir=self.logger_dir,
                                instance=True)

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

    def removeForward(self, other):
        self.forward = [vertex for vertex in self.forward if vertex.vertex_id != other.vertex_id]

    def addBackward(self, other):
        self.backward.append(other)

    def removeBackward(self, other):
        self.backward = [vertex for vertex in self.backward if vertex.vertex_id != other.vertex_id]

    def prune(self):
        vertices = []
        vids = []

        for v in self.forward:
            vertices.append(v)
            vids.append(v.vertex_id)

        n_vid = len(vids)
        i = 0

        while i < n_vid:
            vertex = vertices[i]

            for sub_vertex in vertex.forward:
                if sub_vertex.vertex_id == self.vertex_id:
                    vertex.removeForward(other=self)
                    self.removeBackward(other=vertex)
                    self.logger.debug(f"removeEdge {vertex.vertex_id} -> {self.vertex_id}", extra=self.extra)

                # sub_vertex.vertex_id 不等於 curr_id 且不在 indexs 當中
                elif sub_vertex.vertex_id not in vids:
                    # 加入 indexs 當中，以檢查其子節點是否和 curr_vertex 形成環狀結構
                    vids.append(sub_vertex.vertex_id)
                    vertices.append(sub_vertex)

            n_vid = len(vids)
            i += 1
            self.logger.debug(f"i: {i}, vids({n_vid}): {vids}", extra=self.extra)


class Graph:
    def __init__(self, logger_dir="graph", logger_name=datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")):
        self.logger_dir = logger_dir
        self.logger_name = logger_name
        self.extra = {"className": self.__class__.__name__}
        self.logger = getLogger(logger_name=self.logger_name,
                                to_file=True,
                                time_file=False,
                                file_dir=self.logger_dir,
                                instance=True)

        self.vertices = []
        self.head = []
        self.tail = []

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

    @staticmethod
    def removeEdge(vertex1: Vertex, vertex2: Vertex):
        vertex1.removeForward(vertex2)
        vertex2.removeBackward(vertex1)

    def addEdgeById(self, id1: int, id2: int):
        vertex1 = self.findVertex(vertex_id=id1)
        vertex2 = self.findVertex(vertex_id=id2)

        if vertex1 is not None and vertex2 is not None:
            Graph.addEdge(vertex1=vertex1, vertex2=vertex2)

    def removeEdgeById(self, id1: int, id2: int):
        vertex1 = self.findVertex(vertex_id=id1)
        vertex2 = self.findVertex(vertex_id=id2)

        if vertex1 is not None and vertex2 is not None:
            Graph.removeEdge(vertex1=vertex1, vertex2=vertex2)

    def loadAdjacencyMatrix(self, matrix, n_cell):
        for i in range(n_cell):
            structure_gene = matrix[i]
            connection_indexs = np.where(structure_gene == 1.0)[0]

            if len(connection_indexs) > 0:
                self.addVertexById(vertex_id=i)

                for index in connection_indexs:
                    self.addVertexById(vertex_id=index)
                    self.addEdgeById(id1=i, id2=index)

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

    # 節點 與 邊 添加完後的建構，去除環狀結構，並篩選出頭尾節點
    def build(self):
        self.prune()

        for vertex in self.vertices:
            if len(vertex.backward) == 0:
                self.head.append(vertex)
            elif len(vertex.forward) == 0:
                self.tail.append(vertex)

    # 將 forward 以及 forward 的 forward 當中的親頂點移除
    def prune(self):
        for curr_vertex in self.vertices:
            self.logger.debug(f"curr_vertex: {curr_vertex}", extra=self.extra)
            curr_vertex.prune()


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
    gene = gene.reshape((n_cell, n_cell))
    print(gene)

    graph = Graph()
    graph.loadAdjacencyMatrix(matrix=gene, n_cell=n_cell)
    graph.build()

    for vertex in graph.vertices:
        print(vertex)
