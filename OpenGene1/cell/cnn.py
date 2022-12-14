import datetime
import math

import numpy as np

from OpenGene1.cell import BaseCell
from OpenGene1.gene import translateStruct, Gene


class CnnCell(BaseCell):
    def __init__(self, gene,
                 logger_dir="cell", logger_name=datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")):
        super().__init__(gene=gene, n_struct=4, n_value=49,
                         logger_dir=logger_dir, logger_name=logger_name)

        # 結構基因組
        struct_genome = self.nextStructGenome()

        self.kernel = None

        # kernal size 應保持在奇數 (1, 2, 3, 4) -> (1, 3, 5, 7)
        # kernal 只有二維(長、寬)，因此輸入數據有多少通道，輸出就有多少通道
        # kernal 最大尺寸為 7 * 7 = 49，因此至少需要 49 個數值基因組
        self.h_kernel = translateStruct(next(struct_genome)) * 2 - 1
        self.w_kernel = translateStruct(next(struct_genome)) * 2 - 1

        # stride: (1, 2, 3, 4)
        self.h_stride = translateStruct(next(struct_genome))
        self.w_stride = translateStruct(next(struct_genome))

        self.build()

    @staticmethod
    def getPadding(input_size, kernal_size, stride):
        """
        圖片尺寸 - kernal size 後，應為 stride 的整數倍。
        又 padding 數值應為偶數，使得該維度兩邊添加相同數量的 padding value

        :param input_size: 圖片尺寸
        :param kernal_size: kernal 尺寸
        :param stride: kernal 步長
        :return: 單邊 padding 個數
        """
        padding = stride * (input_size - 1) - input_size + kernal_size

        if padding & 1:
            padding += 1

        return int(padding / 2)

    @staticmethod
    def getNumber(input_size, kernal_size, stride):
        return (input_size - kernal_size) / stride + 1

    def build(self):
        value_genome = self.nextValueGenome()
        kernel_size = self.h_kernel * self.w_kernel
        kernel = []

        for k in range(kernel_size):
            value = Gene.signValue(next(value_genome))
            kernel.append(value)

        self.kernel = np.array(kernel).reshape((self.h_kernel, self.w_kernel))

    def call(self, input_data: np.array):
        c, h, w = input_data.shape
        outputs = np.empty((c, h, w))

        h_padding = CnnCell.getPadding(input_size=h, kernal_size=self.h_kernel, stride=self.h_stride)
        w_padding = CnnCell.getPadding(input_size=w, kernal_size=self.w_kernel, stride=self.w_stride)

        x = np.pad(input_data,
                   pad_width=((0, 0), (h_padding, h_padding), (w_padding, w_padding)),
                   mode='constant',
                   constant_values=0)

        _, height, width = x.shape
        height = height - self.h_kernel + 1
        width = width - self.w_kernel + 1

        for hi, h in enumerate(range(0, height, self.h_stride)):
            for wi, w in enumerate(range(0, width, self.w_stride)):
                output = np.sum(x[:, h: h + self.h_kernel, w: w + self.w_kernel] * self.kernel, axis=(1, 2))
                output = output.reshape((-1, 1, 1))
                outputs[:, hi: hi + 1, wi: wi + 1] = output

        return np.array(outputs)


if __name__ == "__main__":
    gene = CnnCell.createCellGene()
    cnn = CnnCell(gene=gene)
    print(f"(h_kernal, w_kernal) = ({cnn.h_kernel}, {cnn.w_kernel})")
    print(f"(h_stride, w_stride) = ({cnn.h_stride}, {cnn.w_stride})")

    x = np.random.rand(2, 3, 9)
    print("x\n", x)
    output = cnn.call(input_data=x)
    print(f"output.shape = {output.shape}")
    print("output\n", output)
