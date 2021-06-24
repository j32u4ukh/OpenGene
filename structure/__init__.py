import datetime

import numpy as np

from submodule.Xu3.utils import getLogger


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
        self.layer = 0
        self.cell = None
        self.forward = []
        self.backward = []
        self.output = None

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

    # 將 forward 以及 forward 的 forward 當中的親頂點移除
    def prune(self):
        vertices = []
        vids = []

        for v in self.forward:
            vertices.append(v)
            vids.append(v.vertex_id)

        n_vid = len(vids)
        i = 0
        self.logger.debug(f"i: {i}, vids({n_vid}): {vids}", extra=self.extra)

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

    def setLayer(self, layer):
        """
        基本上會是前一層的 layer + 1，但在前一層不只一個的情況下，會取較大的那個，以等待前面的訊號傳播完成

        :param layer:
        :return:
        """
        self.layer = max(self.layer, layer)

    def setCell(self, cell):
        self.cell = cell

    def call(self, input_data: np.array = None):
        if input_data is None:
            last_outputs = []

            for vb in self.backward:
                last_outputs.append(vb.output)

            input_data = combineOutputs(*last_outputs)

        self.logger.debug(f"input: {input_data.shape}", extra=self.extra)
        self.output = self.cell.call(input_data=input_data)
        self.logger.debug(f"output: {self.output.shape}", extra=self.extra)


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

        # 排序 layer 數值
        self.layer = []

        # 管理各個 layer 的節點
        self.layer_vertices = {}

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
                vertices = [vertex]
                n_vertex = len(vertices)
                idx = 0

                while idx < n_vertex:
                    v = vertices[idx]
                    layer = v.layer + 1

                    # 幫自己的下一層設置 layer
                    for vf in v.forward:
                        vf.setLayer(layer=layer)
                        self.logger.debug(f"vertex: {vf}, layer: {vf.layer}", extra=self.extra)
                        vertices.append(vf)

                    n_vertex = len(vertices)
                    idx += 1

        for vertex in self.vertices:
            layer = vertex.layer

            if not self.layer_vertices.__contains__(layer):
                self.layer.append(layer)
                self.layer_vertices[layer] = []

            self.layer_vertices[layer].append(vertex)

        self.layer.sort()

    # 移除環狀結構
    def prune(self):
        for vertex in self.vertices:
            self.logger.debug(f"vertex: {vertex}", extra=self.extra)
            vertex.prune()

    def setCells(self, cells):
        for vertex in self.vertices:
            vid = vertex.vertex_id
            vertex.setCell(cell=cells[vid])

    def call(self, input_data):
        # region Head
        layer_vertices = self.layer_vertices[0]

        for vertex in layer_vertices:
            vertex.call(input_data=input_data)
        # endregion

        # region Body
        n_layer = len(self.layer)

        # 這裡有計算最後一個 layer
        for layer in range(1, n_layer):
            layer_vertices = self.layer_vertices[layer]

            for vertex in layer_vertices:
                vertex.call()
        # endregion

        # region Tail
        layer = self.layer[-1]
        layer_vertices = self.layer_vertices[layer]
        outputs = []

        # 取出最後一個 layer 的輸出
        for vertex in layer_vertices:
            outputs.append(vertex.output)

        # 將多筆輸出合併成最終輸出
        result = combineOutputs(*outputs)
        # endregion

        return result


# TODO: 或許應直接發展網狀結構，但輸入直線型的基因定義，直接退化成直線型
# TODO: 網狀結構類別，用於管理細胞結構的管理
# TODO: 不允許環狀結構，即根節點不能同時作為子節點或孫節點
class Structure:
    def __init__(self, gene: np.array, n_cell: int,
                 logger_dir="structure", logger_name=datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")):
        self.graph = Graph(logger_dir=logger_dir, logger_name=logger_name)
        self.buildStructure(gene=gene, n_cell=n_cell)

    # 根據節構基因，形成 Structure
    def buildStructure(self, gene: np.array, n_cell: int):
        adjacency_matrix = gene.reshape((n_cell, n_cell))
        self.graph.loadAdjacencyMatrix(matrix=adjacency_matrix, n_cell=n_cell)

    def setCells(self, cells):
        self.graph.setCells(cells=cells)

    def call(self, input_data):
        output = self.graph.call(input_data=input_data)

        return output


def createGraphStructure(n_cell, p_rate=0.1):
    gene = np.zeros((n_cell, n_cell))
    p = np.random.random((n_cell, n_cell))
    node_indexs = np.where(p < p_rate)
    gene[node_indexs] = 1.0
    np.fill_diagonal(gene, 0.0)

    return gene.flatten()


def combineOutputs(*outputs):
    n_output = len(outputs)

    if n_output == 0:
        return outputs[0]

    combined_output = None
    c, h, w = 0, 0, 0

    for output in outputs:
        shape = output.shape
        c = max(c, shape[0])
        h = max(h, shape[1])
        w = max(w, shape[2])

    for i in range(n_output):
        output = outputs[i]
        shape = output.shape

        output = np.pad(output,
                        pad_width=((0, 0), (0, h - shape[1]), (0, w - shape[2])),
                        mode='constant',
                        constant_values=0)

        if combined_output is None:
            combined_output = output
        else:
            combined_output = np.concatenate((combined_output, output), axis=0)

    return combined_output


if __name__ == "__main__":
    from cell import ArbitraryCell

    def testGraph1():
        n_cell = 10
        gene = createGraphStructure(n_cell=n_cell, p_rate=0.1)
        gene = gene.reshape((n_cell, n_cell))
        print(gene)

        graph = Graph()
        graph.loadAdjacencyMatrix(matrix=gene, n_cell=n_cell)
        graph.build()

        for vertex in graph.vertices:
            print(vertex)


    def testVertexOneToMulti():
        v1 = Vertex(vertex_id=0)
        v1.setCell(cell=ArbitraryCell(cell=np.random.rand(1, 1, 4)))

        v2 = Vertex(vertex_id=1)
        v2.setCell(cell=ArbitraryCell(cell=np.random.rand(1, 2, 3)))

        v3 = Vertex(vertex_id=2)
        v3.setCell(cell=ArbitraryCell(cell=np.random.rand(1, 3, 2)))

        v1.addForward(v2)
        v2.addBackward(v1)

        v1.addForward(v3)
        v3.addBackward(v1)

        x = np.random.rand(1, 3, 4)
        v1.call(input_data=x)
        print("v1:", v1.output.shape)

        v2.call(input_data=v1.output)
        v3.call(input_data=v1.output)
        print("v2:", v2.output.shape)
        print("v3:", v3.output.shape)

        output = combineOutputs(v2.output, v3.output)
        print("output:", output.shape)


    def testVertexMultiToOne():
        v1 = Vertex(vertex_id=0)
        v1.setCell(cell=ArbitraryCell(cell=np.random.rand(1, 1, 4)))

        v2 = Vertex(vertex_id=1)
        v2.setCell(cell=ArbitraryCell(cell=np.random.rand(1, 2, 3)))

        v3 = Vertex(vertex_id=2)
        v3.setCell(cell=ArbitraryCell(cell=np.random.rand(1, 3, 2)))

        v1.addForward(v3)
        v3.addBackward(v1)

        v2.addForward(v3)
        v3.addBackward(v2)
        print(v3.backward)

        x1 = np.random.rand(1, 3, 4)
        v1.call(input_data=x1)  # (1, 1, 4)

        x2 = np.random.rand(1, 5, 2)
        v2.call(input_data=x2)  # (1, 2, 3)

        # combine (1, 1, 4) with (1, 2, 3) -> (2, 2, 4)
        # x3 = combineOutputs(v1.output, v2.output)
        # v3.call(input_data=x3)
        v3.call()  # (1, 3, 2)


    def testGraph2():
        graph = Graph()

        v1 = Vertex(vertex_id=0)
        v1.setCell(cell=ArbitraryCell(cell=np.random.rand(1, 1, 4)))
        graph.addVertex(v1)

        v2 = Vertex(vertex_id=1)
        v2.setCell(cell=ArbitraryCell(cell=np.random.rand(1, 2, 3)))
        graph.addVertex(v2)

        v3 = Vertex(vertex_id=2)
        v3.setCell(cell=ArbitraryCell(cell=np.random.rand(1, 3, 2)))
        graph.addVertex(v3)

        v4 = Vertex(vertex_id=3)
        v4.setCell(cell=ArbitraryCell(cell=np.random.rand(1, 5, 2)))
        graph.addVertex(v4)

        graph.addEdge(v1, v2)
        graph.addEdge(v2, v3)
        graph.addEdge(v4, v3)

        graph.build()
        print("==================================================")
        output = graph.call(input_data=np.random.rand(1, 3, 5))
        print("output:", output.shape)
