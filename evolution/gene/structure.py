import math

import cv2
import numpy as np

import utils.math as umath

activation_dict = {1: umath.origin,
                   2: umath.relu,
                   3: umath.sigmoid,
                   4: umath.absFunc}


class CellNet:
    def __init__(self):
        self.cells = []

    def __call__(self, x):
        for cell in self.cells:
            x = call(x, weights=cell.weights, bias=cell.bias, output_size=cell.output_size,
                     last_channel=cell.last_channel, channels_array=cell.channels,
                     filter_size=(cell.h_filter, cell.w_filter), window_size=(cell.h_window, cell.w_window),
                     activate_func=cell.activate_func)

        return x

    def compile(self, last_channel=3):
        for cell in self.cells:
            cell.compile(data=[], last_channel=last_channel)
            last_channel = cell.output_size


class Cell:
    def __init__(self, digits=8, steps=8, activation_code=0, filter_size=(2, 2), window_number=(2, 2), output_size=1):
        # 使用幾個數值作為最小定義單位 -> 幾個 0/1 來換算成數值或定義結構
        self.digits = digits

        # 間隔幾個數值再取下一組基因組
        self.steps = steps

        # activation(2): 0.origin 1.relu 2.sigmoid 3.abs(取絕對值)
        self.activate_func = activation_dict[activation_code]

        # 濾波器尺寸
        self.h_filter, self.w_filter = filter_size

        # 濾波器取樣個數(長/寬)
        self.h_window, self.w_window = window_number

        # 輸出個數(n_output)
        self.output_size = output_size

        # 權重
        self.weights = None

        # 偏誤
        self.bias = None

        # 最後一個輸出的深度
        self.last_channel = None

        #
        self.channels = None

        # 8 位元當中，第 1 位數值用於區分正負，剩下則計算實際數值，用於 weights
        self.real_number_multiplier = 2 ** np.arange(self.digits - 1)[::-1] / float(2 ** (self.digits - 1) - 1)

        # 8 位元都用於計算實際數值，都是正數，用於 bias
        self.non_negative_multiplier = 2 ** np.arange(self.digits)[::-1] / float(2 ** self.digits - 1)

    def __call__(self, x):
        outputs = call(x, weights=self.weights, bias=self.bias, output_size=self.output_size,
                       last_channel=self.last_channel, channels_array=self.channels,
                       filter_size=(self.h_filter, self.w_filter), window_size=(self.h_window, self.w_window),
                       activate_func=self.activate_func)

        # input_data, (h_stride, w_stride) = inputReconstruct(x,
        #                                                     filter_shape=(self.h_filter, self.w_filter),
        #                                                     window=(self.h_window, self.w_window))
        #
        # output = []
        # for h in range(self.h_window):
        #     for w in range(self.w_window):
        #         h_index = h * h_stride
        #         w_index = w * w_stride
        #         patch = input_data[:, h_index: h_index + self.h_filter, w_index: w_index + self.w_filter]
        #         result = self.activate_func(patch * self.weights + self.bias)
        #         result = result.sum(axis=0)
        #         output.append(result)
        #
        # output = np.array(output)
        # outputs = []
        #
        # for out in range(self.output_size):
        #     channels = self.channels[out]
        #     channels = channels.reshape((self.last_channel, 1, 1))
        #     channel = (output * channels).sum(axis=0)
        #     outputs.append(channel)
        #
        # outputs = np.array(outputs)

        return outputs

    def compile(self, data, last_channel=16):
        """
        ex: digits = 8, steps = 4
        ********
            ********
                ********

        :param data:
        :param last_channel:
        :return:
        """
        temp_weights = []
        temp_bias = []

        # 一個 filter 占用的基因個數
        filter_size = self.digits + (self.h_filter * self.w_filter - 1) * self.steps

        # 若 self.steps = self.digits，則 bias 從索引值 filter_size 開始取值
        bias_start_index = filter_size - self.digits + self.steps

        for i in range(0, filter_size - self.steps, self.steps):
            weights_gene = data[i: i + self.digits]
            bias_gene = data[i + bias_start_index: i + bias_start_index + self.digits]
            print(f"bias: ({i + bias_start_index}, {i + bias_start_index + self.digits})")

            weights = (weights_gene[1:] * self.real_number_multiplier).sum()
            bias = (bias_gene * self.non_negative_multiplier).sum()

            if weights_gene[0] == 0:
                weights *= -1

            temp_weights.append(weights)
            temp_bias.append(bias)

        # temp_weights: 數值個數與 self.weights 相同的一維陣列，此處再將它 reshape 為 (self.h_filter, self.w_filter)
        self.weights = np.array(temp_weights).reshape((self.h_filter, self.w_filter))

        # temp_bias: 數值個數與 self.bias 相同的一維陣列，此處再將它 reshape 為 (self.h_filter, self.w_filter)
        self.bias = np.array(temp_bias).reshape((self.h_filter, self.w_filter))

        # hw 層使用的基因個數
        weights_bias_size = self.digits + (self.h_filter * self.w_filter * 2 - 1) * self.steps

        # 考慮到 hw 層取值範圍 和 channel 層取值會有重疊的部分
        # 若 self.step = self.digits，則 channel 從索引值 weights_bias_size 開始取值
        channel_start_index = weights_bias_size - self.digits + self.steps

        # 前一個 cell 的輸出數(last_channel) 乘上 hw 層的輸出個數，為每個 channel 所需要的數值個數

        # 對於每一層輸入，都會產出 self.h_window * self.w_window 個 hw 層的輸出，
        # last_channel 層輸入就會有 last_channel * self.h_window * self.w_window = self.last_channel 個 hw 層的輸出
        self.last_channel = last_channel * self.h_window * self.w_window
        print(f"self.last_channel = {last_channel} * {self.h_window} * {self.w_window} = {self.last_channel}")

        # 考慮 self.step 的情況下，定義一個 output_channel 所需的位數，當 self.step = self.digits 時，會和不重疊取值時相同
        # 定義 channel 的一個加權比例值，會需要 out_size 個基因。(每一層輸出有 self.last_channel 個加權比例值)
        out_size = self.digits + (self.last_channel - 1) * self.steps
        print(f"out_size: {out_size}, self.output_size: {self.output_size}")
        output_channels = []

        # self.output_size: 最終輸出層數
        for out in range(self.output_size):
            channels = []

            for c in range(self.last_channel):
                c_start = channel_start_index + out * out_size + c * self.steps

                # 當 self.step = self.digits 時，會和不重疊取值時相同
                c_stop = channel_start_index + out * out_size + c * self.steps + self.digits
                print(f"c_start: {c_start}, c_stop: {c_stop}")

                # 根據 c_start, c_stop 從 data 中取得 channel 加權基因組
                channel_gene = data[c_start: c_stop]

                # '加權基因組'利用 self.non_negative_multiplier 轉換成數值
                channel = (channel_gene * self.non_negative_multiplier).sum()

                # 取得 channel No.c 的加權比例值
                channels.append(channel)

            channels = np.array(channels)

            # 各 channel 的加權比例值，除以總合，確保比例加總為 1
            output_channels.append(channels / channels.sum())

        # self.output_size 組'各 channel 的加權比例值'
        self.channels = np.array(output_channels).reshape((self.output_size, -1))


# 產生 n_gene 個 0/1 基因
def createGenome(n_gene):
    gene = np.random.randint(low=0, high=2, size=n_gene)
    return gene


# 轉譯架構類基因(不同於定義數值的基因)
def translateStruct(gene: np.array):
    struct_multiplier = np.array([2, 1])
    index = (gene * struct_multiplier).sum() + 1

    return index


# 將 input_data 縮放成當前 filter & window 所需要的維度
def inputReconstruct(input_data, filter_shape=(2, 2), window=(2, 2)):
    input_dim, height, width = input_data.shape
    f_height, f_width = filter_shape
    win_height, win_width = window

    # reconstructParam: 為了'當前 filter & window 所需要的維度'，所需要縮放的比例和 padding 的大小，以及 stride 大小
    pad_height, resize_height, stride_height = reconstructParam(input_size=height,
                                                                filter_size=f_height,
                                                                window_size=win_height)
    pad_width, resize_width, stride_width = reconstructParam(input_size=width,
                                                             filter_size=f_width,
                                                             window_size=win_width)

    if resize_height != height or resize_width != width:
        result = []

        # 沿著深度，對數據做縮放，再存回 result 當中
        for data in input_data:
            # cv2.resize 的 height, width 和一般的順序相反
            # cv2.resize 返回值是對複製的數據做 resize
            res = cv2.resize(data, (resize_width, resize_height))
            result.append(res)

        result = np.array(result)
    else:
        # 無須縮放，複製一份 input_data
        result = input_data.copy()

    if pad_height != 0 or pad_width != 0:
        result = np.pad(result, ((0, 0), (pad_height, pad_height), (pad_width, pad_width)), 'constant')

    return result, (stride_height, stride_width)


# 計算為了'縮放成當前 filter & window 所需要的維度'，所需要縮放的比例和 padding 的大小，以及 stride 大小
def reconstructParam(input_size, filter_size, window_size):
    boundary = (filter_size + window_size - 1, filter_size * window_size)

    # 根據 input_data 大小，以及 filter 所需要的大小，決定 strides, padding
    if input_size < boundary[0]:
        # 原始尺寸
        resize_size = input_size

        # 填充至最小要求大小
        pad_size = math.ceil((boundary[0] - input_size) / 2)

        stride_size = int((input_size + 2 * pad_size - filter_size) / (window_size - 1))

    elif boundary[1] < input_size:
        # 縮放至最大容許大小
        resize_size = boundary[1]

        # 無填充
        pad_size = 0

        if window_size == 1:
            stride_size = 0
        else:
            stride_size = filter_size
    else:
        # 原始尺寸
        resize_size = input_size

        # 無填充
        pad_size = 0

        if window_size == 1:
            stride_size = 0
        else:
            stride_size = (input_size - filter_size) / (window_size - 1)

    return pad_size, resize_size, math.floor(stride_size)


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


def circleData(full, start_index, length):
    """
    讓 numpy 數列形成一個環，讓取值不會不足

    :param full: 全部數值
    :param start_index: 起始索引值
    :param length: 要求長度
    :return:
    """
    n_number = len(full)

    if n_number - start_index >= length:
        result = full[np.array(np.arange(start_index, start_index + length))]
        next_index = (start_index + length) % n_number
    else:
        result = full[start_index:]

        leftover_round, leftover_number = divmod(length - (n_number - start_index), n_number)

        for r in range(leftover_round):
            result = np.hstack([result, full])

        result = np.hstack([result, full[:leftover_number]])
        next_index = leftover_number

    return next_index, result


if __name__ == "__main__":
    """
    # 這兩種 cell 組成一組，避免維度爆炸
    activation(2): 0.origin 1.relu 2.sigmoid 3.abs(取絕對值) 
    cell type(1): 0.wh-oriented 1.channel-oriented
    -----
    wh-oriented
    filter size(2*2=4): height 1 ~ 4, width 1 ~ 4 
    window size(2*2=4): horizontal 1 ~ 4, vertical 1 ~ 4 
    weights(16*8=128): digits = 8
    (1) 正負 (7)[0.50393701, 0.2519685 , 0.12598425, 0.06299213, 0.03149606, 0.01574803, 0.00787402] (16)cell 個數
    bias(16*8=128): digits = 8
    (8)[0.50196078, 0.25098039, 0.1254902 , 0.0627451 , 0.03137255, 0.01568627, 0.00784314, 0.00392157] (16)cell 個數
    -----
    '輸入層'經 weights & bias & activation(wh 層) 的計算後，稱之為'中間層'。
    '中間層'透過'比例加權層'加權後，產生 output_size 個輸出層
    -----
    channel-oriented
    weights(input channel * digits): 每個 input channel 對應一個比例參數，再將各個 input channel 加總
    input channel: 一次 wh-oriented 後最多會有 16 個 channel -> (16*8=128)
    output channel(2): 1 ~ 4，決定上述輸出有幾組

    # temp = np.random.rand(5, 4, 3)
    # scale = np.random.rand(5, 1, 1)
    -----
    reconstructParam & inputReconstruct: 處理輸入的數據維度和所需大小不同的問題
    令 n_layer = 4 -> 利用 30 個位數來定義過程中 cell 的大小，每個 cell 需要 12 個位數來定義，下一個偏移 6 位後再取出 12 個。
    weights、bias、channel: 最多都會需要 128 個位數來定義，利用偏移一半長度來減少所需位數，最多需有 1600 位數。
    綜合: 30 + 1600 = 1630
    """
    # last_channel: 輸入數據深度
    last_channel = 1
    input_data = np.random.rand(last_channel, 4, 4)

    # struct_gene: 用於定義'結構'的基因組
    struct_gene = createGenome(n_gene=12)

    # 激勵函數的代碼
    activation_code = translateStruct(struct_gene[0: 2])

    # 濾波器大小
    filter_size = (translateStruct(struct_gene[2: 4]), translateStruct(struct_gene[4: 6]))

    # 滑動窗口個數
    window_number = (translateStruct(struct_gene[6: 8]), translateStruct(struct_gene[8: 10]))

    # 輸出數據深度
    output_size = translateStruct(struct_gene[10: 12])

    cell = Cell(digits=8,
                steps=4,
                activation_code=activation_code,
                filter_size=filter_size,
                window_number=window_number,
                output_size=output_size)

    # max n_gene = 4 * 4 * 8 * 2 + 4 * 4 * 64 * 4 * 8 = 33024
    n_gene = (cell.h_filter * cell.w_filter * cell.digits * 2 +
              cell.h_window * cell.w_window * last_channel * cell.output_size * cell.digits)
    # n_gene = cell.digits + (cell.h_filter * cell.w_filter * 2 + cell.last_channel * cell.output_size - 1) * cell.steps

    # value_gene: 用於定義'結構當中數值'的基因組
    value_gene = createGenome(n_gene=n_gene)

    # 根據 value_gene 編譯，設置 cell 當中的數值
    cell.compile(data=value_gene, last_channel=last_channel)

    # input_data 通過 cell 的運算，獲得 outputs
    outputs = cell(input_data)
    print(f"outputs.shape: {outputs.shape}, cell.filter: ({cell.h_filter}, {cell.w_filter}), "
          f"cell.output_size: {cell.output_size}")
    print("#Value gene:", cell.digits + (cell.h_filter * cell.w_filter * 2 +
                                         cell.last_channel * cell.output_size - 1) * cell.steps)
