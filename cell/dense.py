import datetime
import math

import numpy as np

import utils.math as umath
from cell import BaseCell
from gene import Gene, createGene, translateStruct


class DenseCell(BaseCell):
    n_gene = 208
    activation_dict = {1: umath.origin,
                       2: umath.relu,
                       3: umath.sigmoid,
                       4: umath.absFunc}

    def __init__(self, gene,
                 logger_dir="dense_cell", logger_name=datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")):
        super().__init__(gene=gene, n_struct=6, n_value=48,
                         logger_dir=logger_dir, logger_name=logger_name)

        struct_genome = self.nextStructGenome()

        # activation(2): 1.origin 2.relu 3.sigmoid 4.abs(取絕對值)
        activation_code = translateStruct(next(struct_genome))
        self.activate_func = DenseCell.activation_dict[activation_code]

        # 濾波器尺寸
        self.h_filter = translateStruct(next(struct_genome))
        self.w_filter = translateStruct(next(struct_genome))
        self.logger.debug(f"filter: ({self.h_filter}, {self.w_filter})", extra=self.extra)

        # 濾波器取樣個數(長/寬)
        self.h_window = translateStruct(next(struct_genome))
        self.w_window = translateStruct(next(struct_genome))
        self.logger.debug(f"window: ({self.h_window}, {self.w_window})", extra=self.extra)

        # 輸出個數(n_output)
        self.output_size = translateStruct(next(struct_genome))
        self.logger.debug(f"output_size: {self.output_size}", extra=self.extra)

        # 權重
        self.weights = None

        # 偏誤
        self.bias = None

        #
        self.channels = None

        self.build()

    # 計算為了'縮放成當前 filter & window 所需要的維度'，所需要縮放的比例和 padding 的大小，以及 stride 大小
    @staticmethod
    def reconstructParam(input_size, filter_size, n_window):
        """
        根據 input_size，以及 filter_size，決定 strides, padding

        :param input_size: 輸入數據大小
        :param filter_size: 濾波器大小
        :param n_window: 視窗個數
        :return:
        """
        min_require = filter_size + n_window - 1

        if input_size < min_require:

            # 填充至最小要求大小
            pad_size = math.ceil((min_require - input_size) / 2)

        else:
            # 無填充
            pad_size = 0

        if n_window == 1:
            stride_size = 0

        else:
            stride_size = math.floor((input_size + 2 * pad_size - filter_size) / (n_window - 1))

        return pad_size, stride_size

    def build(self):
        value_genome = self.nextValueGenome()

        temp_weights = []
        temp_bias = []

        for _ in range(self.h_filter * self.w_filter):
            weights_gene = next(value_genome)
            bias_gene = next(value_genome)

            weights = Gene.signValue(weights_gene)
            bias = Gene.unsignValue(bias_gene)

            temp_weights.append(weights)
            temp_bias.append(bias)

        # temp_weights: 數值個數與 self.weights 相同的一維陣列，此處再將它 reshape 為 (self.h_filter, self.w_filter)
        self.weights = np.array(temp_weights).reshape((self.h_filter, self.w_filter))

        # temp_bias: 數值個數與 self.bias 相同的一維陣列，此處再將它 reshape 為 (self.h_filter, self.w_filter)
        self.bias = np.array(temp_bias).reshape((self.h_filter, self.w_filter))

        channels = []

        # self.output_size: 最終輸出層數
        for _ in range(self.output_size):
            temp_channels = []

            for _ in range(self.output_size):
                channel_gene = next(value_genome)

                # 都會是 正值 或是 0
                channel = Gene.unsignValue(channel_gene)

                temp_channels.append(channel)

            channels.append(temp_channels)

        # self.output_size 組'各 channel 的加權比例值'
        self.channels = np.array(channels).reshape((self.output_size, self.output_size))

    def call(self, input_data):
        input_data, (h_stride, w_stride) = self.inputReconstruct(input_data)
        self.logger.debug(f"input_data: {input_data.shape}, stride: ({h_stride}, {w_stride})", extra=self.extra)

        output = None
        shape = input_data.shape
        ndim = len(shape)
        n_channel = min(shape[-3], self.output_size)
        c_axis, h_axis, w_axis = ndim - 3, ndim - 2, ndim - 1
        self.logger.debug(f"n_channel: {n_channel}", extra=self.extra)

        for h in range(self.h_window):
            temp_output = None
            self.logger.debug(f"h: {h}", extra=self.extra)

            for w in range(self.w_window):
                self.logger.debug(f"(h, w): ({h}, {w})", extra=self.extra)

                h_index = h * h_stride
                w_index = w * w_stride

                # 根據 h_index, w_index, h_filter, w_filter 取出要運算的區塊
                # 之後加權只使用前 n_channel 個 channel，這裡直接取前 n_channel 個，以減少不必要的計算
                self.logger.debug(f"[{n_channel}, {h_index}: {h_index + self.h_filter}, "
                                  f"{w_index}: {w_index + self.w_filter}]", extra=self.extra)
                patch = input_data[:n_channel, h_index: h_index + self.h_filter, w_index: w_index + self.w_filter]
                self.logger.debug(f"patch.shape: {patch.shape}", extra=self.extra)

                # 運算後結果通過非線性的激勵函數
                result = self.activate_func(patch * self.weights + self.bias)
                self.logger.debug(f"result.shape: {result.shape}", extra=self.extra)

                # 根據不同 channel 參數對 result 進行加權
                out = None
                for o in range(self.output_size):
                    self.logger.debug(f"(o, h, w): ({o}, {h}, {w})", extra=self.extra)

                    channels = self.channels[o, :n_channel]
                    denominator = channels.sum()

                    if denominator == 0:
                        # 加總數值是 0，只在原始 channels 數值都為 0 才有可能，
                        # 但也表示這些 channel 權重雖然都相同但數值應該要很低，之後不同 n_channel 取值時不會佔太高的權重
                        val = 1.0 / Gene.getUnsignNormalizer()
                        self.channels[o, :n_channel] = np.array([val for _ in range(n_channel)])
                        denominator = self.channels[o, :n_channel].sum()

                    channels /= denominator
                    channels = channels.reshape((n_channel, 1, 1))
                    temp = (channels * result).sum(axis=c_axis)
                    temp = temp.reshape((1, self.h_filter, self.w_filter))
                    self.logger.debug(f"temp.shape: {temp.shape}", extra=self.extra)

                    if out is None:
                        out = temp
                    else:
                        out = np.concatenate((out, temp), axis=c_axis)

                    self.logger.debug(f"out.shape: {out.shape}", extra=self.extra)

                if temp_output is None:
                    temp_output = out
                else:
                    temp_output = np.concatenate((temp_output, out), axis=w_axis)

                self.logger.debug(f"temp_output.shape: {temp_output.shape}", extra=self.extra)

            if output is None:
                output = temp_output
            else:
                output = np.concatenate((output, temp_output), axis=h_axis)

            self.logger.debug(f"output.shape: {output.shape}", extra=self.extra)

        return output

    # 將 input_data 縮放成當前 filter & window 所需要的維度
    def inputReconstruct(self, input_data):
        _, height, width = input_data.shape

        # reconstructParam: 為了'當前 filter & window 所需要的維度'，所需要縮放的比例和 padding 的大小，以及 stride 大小
        pad_height, stride_height = self.reconstructParam(input_size=height,
                                                          filter_size=self.h_filter,
                                                          n_window=self.h_window)
        pad_width, stride_width = self.reconstructParam(input_size=width,
                                                        filter_size=self.w_filter,
                                                        n_window=self.w_window)

        result = input_data.copy()

        if pad_height != 0 or pad_width != 0:
            result = np.pad(result, ((0, 0), (pad_height, pad_height), (pad_width, pad_width)), 'constant')

        return result, (stride_height, stride_width)


if __name__ == "__main__":
    def testDenseCell():
        gene1 = createGene(n_gene=208)
        dense1 = DenseCell(gene1)

        gene2 = createGene(n_gene=208)
        dense2 = DenseCell(gene2)

        input_data = np.random.random((1, 3, 4))
        output1 = dense1.call(input_data)
        print(f"output1: {output1.shape}")

        output2 = dense2.call(output1)
        print(f"output2: {output2.shape}")
