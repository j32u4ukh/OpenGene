import datetime
import math
from cell import BaseCell
from gene import translateStruct
import numpy as np


class CnnCell(BaseCell):
    def __init__(self, gene,
                 logger_dir="cell", logger_name=datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")):
        super().__init__(gene=gene, n_struct=4, n_value=49,
                         logger_dir=logger_dir, logger_name=logger_name)

        # 結構基因組
        struct_genome = self.nextStructGenome()

        self.kernal = None

        # kernal size 應保持在奇數 (1, 2, 3, 4) -> (1, 3, 5, 7)
        # kernal 只有二維(長、寬)，因此輸入數據有多少通道，輸出就有多少通道
        # kernal 最大尺寸為 7 * 7 = 49，因此至少需要 49 個數值基因組
        self.h_kernal = translateStruct(next(struct_genome)) * 2 - 1
        self.w_kernal = translateStruct(next(struct_genome)) * 2 - 1

        # stride: (1, 2, 3, 4)
        self.h_stride = translateStruct(next(struct_genome))
        self.w_stride = translateStruct(next(struct_genome))

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
        # 使用 abs 以確保圖片尺寸比 kernal 尺寸小的時候，也能正常運作
        padding = math.ceil(abs(input_size - kernal_size) / stride)

        if padding & 1:
            padding += 1

        return padding / 2

    @staticmethod
    def getNumber(input_size, kernal_size, stride):
        return (input_size - kernal_size) / stride + 1

    def build(self):
        value_genome = self.nextValueGenome()
        kernal_size = self.h_kernal * self.w_kernal
        kernal = []

        for k in range(kernal_size):
            value = next(value_genome)
            kernal.append(value)

        self.kernal = np.array(kernal).reshape((self.h_kernal, self.w_kernal))

    def call(self, input_data:np.array):
        _, h, w = input_data.shape
        h_padding = CnnCell.getPadding(input_size=h, kernal_size=self.h_kernal, stride=self.h_stride)
        w_padding = CnnCell.getPadding(input_size=w, kernal_size=self.w_kernal, stride=self.w_stride)
        x = np.pad(input_data,
                   pad_width=((0, 0), (h_padding, h_padding), (w_padding, w_padding)),
                   mode='constant',
                   constant_values=0.0)

        _, height, width = x.shape
        h_number = (height - self.h_kernal) / self.h_stride + 1
        w_number = (width - self.w_kernal) / self.w_stride + 1

        for h in range(h_number):
            for w in range(w_number):
                pass

        """
        outputs = np.empty(shape=(out_height, out_width))
        for r, y in enumerate(range(0, padded_inputs.shape[0]-ks[1]+1, stride)):
            for c, x in enumerate(range(0, padded_inputs.shape[1]-ks[0]+1, stride)):
                outputs[r][c] = np.sum(padded_inputs[y:y+ks[1], x:x+ks[0], :] * kernel)
        return outputs
        """

