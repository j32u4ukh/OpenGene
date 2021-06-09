import datetime
import math
from abc import ABCMeta, abstractmethod

import numpy as np

import utils.math as umath
from gene import Gene, createGene, translateStruct
from submodule.Xu3.utils import getLogger


class Cell(metaclass=ABCMeta):
    """
    Cell 如何解讀傳入的基因段，可以根據不同類型的 Cell 有不同的定義。

    @abstractmethod -> 定義要子物件實作的函式
    """
    # 定義結構的基因組個數
    n_struct = None

    # 定義數值的基因組個數
    n_value = None

    def __init__(self, gene, n_struct, n_value, logger_dir="cell",
                 logger_name=datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")):
        """

        :param gene: 傳入的基因段
        :param n_struct: 定義結構的基因組個數
        :param n_value: 定義數值的基因組個數
        """
        self.logger_dir = logger_dir
        self.logger_name = logger_name
        self.extra = {"className": self.__class__.__name__}
        self.logger = getLogger(logger_name=self.logger_name,
                                to_file=True,
                                time_file=False,
                                file_dir=self.logger_dir,
                                instance=True)

        self.index = 0
        self.gene = gene

        self.__class__.n_struct = n_struct
        self.__class__.n_value = n_value

    @classmethod
    def getGeneDemand(cls):
        """
        不同類型的細胞，會需要不同的 n_struct 和 n_value。


        :return: 建構這個細胞所需的基因數量
        """
        gene_demand = cls.n_struct * Gene.struct_digits + (cls.n_value - 1) * Gene.value_steps + Gene.value_digits

        return gene_demand

    @abstractmethod
    def build(self):
        pass

    @abstractmethod
    def call(self, input_data):
        pass

    def nextStructGenome(self):
        for _ in range(self.__class__.n_struct):
            genome = self.gene[self.index: self.index + Gene.struct_digits]
            self.index += Gene.struct_digits

            yield genome

    def nextValueGenome(self):
        """

        :return: 返回長度為 Gene.digits 的基因組
        """
        for _ in range(self.__class__.n_value):
            genome = self.gene[self.index: self.index + Gene.value_digits]
            self.index += Gene.value_steps

            yield genome


class DenseCell(Cell):
    activation_dict = {1: umath.origin,
                       2: umath.relu,
                       3: umath.sigmoid,
                       4: umath.absFunc}

    def __init__(self, gene):
        super().__init__(gene, n_struct=6, n_value=48,
                         logger_dir="DenseCell", logger_name=datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))

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
                channel = Gene.unsignValue(channel_gene)
                temp_channels.append(channel)

            channels.append(temp_channels)

        # self.output_size 組'各 channel 的加權比例值'
        self.channels = np.array(channels).reshape((self.output_size, self.output_size))
        # self.channels /= self.channels.sum()

    def call(self, input_data):
        input_data, (h_stride, w_stride) = inputReconstruct(input_data,
                                                            filter_shape=(self.h_filter, self.w_filter),
                                                            window=(self.h_window, self.w_window))
        output = None
        shape = input_data.shape
        ndim = len(shape)
        n_channel = min(shape[-2], self.output_size)

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
                    channels /= channels.sum()
                    channels = channels.reshape((n_channel, 1, 1))
                    temp = channels * result

                    if out is None:
                        out = temp
                    else:
                        out = np.concatenate((out, temp), axis=ndim - 3)

                    self.logger.debug(f"out.shape: {out.shape}", extra=self.extra)

                if temp_output is None:
                    temp_output = out
                else:
                    temp_output = np.concatenate((temp_output, out), axis=ndim - 1)

                self.logger.debug(f"temp_output.shape: {temp_output.shape}", extra=self.extra)

            if output is None:
                output = temp_output
            else:
                output = np.concatenate((output, temp_output), axis=ndim - 2)

            self.logger.debug(f"output.shape: {output.shape}", extra=self.extra)

        return output


def call(x, weights, bias, output_size, last_channel, channels_array: list,
         filter_size=(2, 2), window_size=(2, 2), activate_func=umath.origin):
    """

    :param x:
    :param weights:
    :param bias:
    :param output_size:
    :param last_channel:
    :param channels_array: 用於加權中間層(weight-bias層的輸出)的加權比例值
    :param filter_size:
    :param window_size:
    :param activate_func:
    :return:
    """
    h_filter, w_filter = filter_size
    h_window, w_window = window_size

    input_data, (h_stride, w_stride) = inputReconstruct(x,
                                                        filter_shape=(h_filter, w_filter),
                                                        window=(h_window, w_window))
    output = []

    for h in range(h_window):
        for w in range(w_window):
            h_index = h * h_stride
            w_index = w * w_stride

            # 根據 h_index, w_index, h_filter, w_filter 取出要運算的區塊
            patch = input_data[:, h_index: h_index + h_filter, w_index: w_index + w_filter]

            # 運算後結果通過非線性的激勵函數
            result = activate_func(patch * weights + bias)

            # 區塊運算結果加總
            result = result.sum(axis=0)
            print(f"[call] result.shape: {result.shape}")

            # 運算加總結果加入 output
            output.append(result)

    output = np.array(output)
    print(f"[call] output.shape: {output.shape}")
    outputs = []

    print(f"[call] output_size: {output_size}, #channels_array: {len(channels_array)}")
    for out in range(output_size):
        # (#channels) = last_channel
        channels = channels_array[out]

        # 各深度的加權比例
        channels = channels.reshape((last_channel, 1, 1))
        channel = (output * channels).sum(axis=0)
        outputs.append(channel)

    outputs = np.array(outputs)

    return outputs


# 將 input_data 縮放成當前 filter & window 所需要的維度
def inputReconstruct(input_data, filter_shape=(2, 2), window=(2, 2)):
    input_dim, height, width = input_data.shape
    f_height, f_width = filter_shape
    win_height, win_width = window

    # reconstructParam: 為了'當前 filter & window 所需要的維度'，所需要縮放的比例和 padding 的大小，以及 stride 大小
    pad_height, stride_height = reconstructParam(input_size=height,
                                                 filter_size=f_height,
                                                 n_window=win_height)
    pad_width, stride_width = reconstructParam(input_size=width,
                                               filter_size=f_width,
                                               n_window=win_width)

    result = input_data.copy()

    if pad_height != 0 or pad_width != 0:
        result = np.pad(result, ((0, 0), (pad_height, pad_height), (pad_width, pad_width)), 'constant')

    return result, (stride_height, stride_width)


# TODO: 排除 stride_size 可能為負數的情況
# 計算為了'縮放成當前 filter & window 所需要的維度'，所需要縮放的比例和 padding 的大小，以及 stride 大小
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
        stride_size = math.floor((input_size - filter_size) / (n_window - 1))

    return pad_size, stride_size


if __name__ == "__main__":
    def testCell():
        gene = createGene(n_gene=24)
        cell = Cell(gene, n_struct=2, n_value=3)

        struct_genome = cell.nextStructGenome()
        print("nextStructGenome:", next(struct_genome))
        print("nextStructGenome:", next(struct_genome))

        value_genome = cell.nextValueGenome()
        print("nextValueGenome:", next(value_genome))
        print("nextValueGenome:", next(value_genome))
        print("nextValueGenome:", next(value_genome))


    gene = createGene(n_gene=208)
    dense = DenseCell(gene)

    input_data = np.random.random((1, 3, 4))
    dense.call(input_data)
