import numpy as np

from cell import DenseCell, call
from gene import createGene, translateStruct

"""
結構一開始提供直線型串接，最終應提供網狀結構，'直線型'為'網狀結構'的退化型。

TODO: 產生直線型串接的基因組
TODO: 網狀結構類別，用於管理細胞結構的管理，某些結構可能造成無窮迴圈(或許容許無窮迴圈的結構存在，但利用每次傳遞訊號強度都會衰退，
讓訊號的傳遞可以自然的傳遞或衰退)
"""


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
    struct_gene = createGene(n_gene=12)

    # 激勵函數的代碼
    activation_code = translateStruct(struct_gene[0: 2])

    # 濾波器大小
    filter_size = (translateStruct(struct_gene[2: 4]), translateStruct(struct_gene[4: 6]))

    # 滑動窗口個數
    window_number = (translateStruct(struct_gene[6: 8]), translateStruct(struct_gene[8: 10]))

    # 輸出數據深度
    output_size = translateStruct(struct_gene[10: 12])

    cell = DenseCell(digits=8,
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
    value_gene = createGene(n_gene=n_gene)

    # 根據 value_gene 編譯，設置 cell 當中的數值
    cell.compile(data=value_gene, last_channel=last_channel)

    # input_data 通過 cell 的運算，獲得 outputs
    outputs = cell(input_data)
    print(f"outputs.shape: {outputs.shape}, cell.filter: ({cell.h_filter}, {cell.w_filter}), "
          f"cell.output_size: {cell.output_size}")
    print("#Value gene:", cell.digits + (cell.h_filter * cell.w_filter * 2 +
                                         cell.last_channel * cell.output_size - 1) * cell.steps)
