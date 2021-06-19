from cell import RawCell, DenseCell
from gene import translateStruct


# 細胞工廠，根據基因定義，返回相對應的細胞
class CellFactory:
    code_book = None

    @staticmethod
    def initCodeBook():
        if CellFactory.code_book is None:
            # TODO: 是否可以改寫成文件形式?而非直接寫在原始碼
            # 參考1: https://stackoverflow.com/questions/553784/can-you-use-a-string-to-instantiate-a-class
            # 參考2: https://stackoverflow.com/questions/452969/does-python-have-an-equivalent-to-java-class-forname
            CellFactory.code_book = {
                0: RawCell,
                1: DenseCell
            }

    @staticmethod
    def createCell(gene):
        code = gene[:8]
        value_gene = gene[8:]
        n_kind = len(CellFactory.code_book)
        code_value = (translateStruct(code) - 1) % n_kind

        cell = CellFactory.code_book[code_value](value_gene)
        return cell


if __name__ == "__main__":
    def testCreateCell():
        from gene import createGene
        import numpy as np
        from cell import Cell

        CellFactory.initCodeBook()
        gene1 = createGene(n_gene=Cell.n_gene)
        code = gene1[:8]
        n_kind = len(CellFactory.code_book)
        code_value = (translateStruct(code) - 1) % n_kind
        print(f"code_value: {code_value}, code: {code}")
        cell1 = CellFactory.createCell(gene1)

        input_data = np.random.random((1, 3, 4))
        output1 = cell1.call(input_data)
        print(f"output1: {output1.shape}")


    testCreateCell()
