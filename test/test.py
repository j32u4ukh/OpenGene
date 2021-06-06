import math

import cv2
import numpy as np
from matplotlib import pyplot as plt


def createGenome(length, digits=8, kind="str"):
    if kind == "str":
        length = (length // digits) + 1
        high = pow(2, digits)
        str_format = "{0:0" + str(digits) + "b}"
        gene = np.random.randint(low=0, high=high, size=length)
        str_gene = ""

        for g in gene:
            str_gene += str_format.format(g)

        return str_gene
    else:
        gene = np.random.randint(low=0, high=2, size=length)
        return gene


length = 100
digits = 8
gene = createGenome(length=length, digits=digits)
n_gene = len(gene)
print("#gene:", n_gene)
print(gene)

value = []

for i in range(0, n_gene - digits + 1, 4):
    temp = gene[i: i + digits]
    print(i, i + digits)
    val = int(temp, 2)
    value.append(val)

n_value = len(value)

x = value[:-1]
y = value[1:]
plt.scatter(x, y)
plt.show()

print(np.corrcoef([x, y]))

digits = 8
# [0.50393701, 0.2519685 , 0.12598425, 0.06299213, 0.03149606, 0.01574803, 0.00787402]
real_number_multiplier = 2 ** np.arange(digits - 1)[::-1] / float(2 ** (digits - 1) - 1)

# [0.50196078, 0.25098039, 0.1254902 , 0.0627451 , 0.03137255, 0.01568627, 0.00784314, 0.00392157]
non_negative_multiplier = 2 ** np.arange(digits)[::-1] / float(2 ** digits - 1)

ohlc = [[29.1, 30.1, 28.1, 31.1],
        [29.2, 30.2, 28.2, 31.2],
        [29.3, 30.3, 28.3, 31.3],
        [29.4, 30.4, 28.4, 31.4],
        [29.5, 30.5, 28.5, 31.5]]

# np_ohlc.shape: (5, 4) -> (5, 1, 4)
np_ohlc = np.array(ohlc)[:, np.newaxis, :]

# pad_ohlc.shape: (5, 4, 4) -> (n_dim, height, width)
pad_ohlc = np.pad(np_ohlc, ((0, 0), (0, 3), (0, 0)), "edge")


# region 處理輸入的數據維度和所需大小不同的問題
def reconstructParam(input_size, filter_size, window_size):
    boundary = (filter_size + window_size - 1, filter_size * window_size)

    # 根據 input_data 大小，以及 filter 所需要的大小，決定 strides, padding
    if input_size < boundary[0]:
        # 原始尺寸
        resize_size = input_size

        # 填充至最小要求大小
        pad_size = math.ceil((boundary[0] - input_size) / 2)

        # stride_size = int((input_size + 2 * pad_size - filter_size) / (window_size - 1))

    elif boundary[1] < input_size:
        # 縮放至最大容許大小
        resize_size = boundary[1]

        # 無填充
        pad_size = 0

        # if window_size == 1:
        #     stride_size = 0
        # else:
        #     stride_size = filter_size
    else:
        # 原始尺寸
        resize_size = input_size

        # 無填充
        pad_size = 0

        # if window_size == 1:
        #     stride_size = 0
        # else:
        #     stride_size = (input_size - filter_size) / (window_size - 1)

    return pad_size, resize_size


def inputReconstruct(input_data, filter_shape=(2, 2), window=(2, 2)):
    input_dim, height, width = input_data.shape
    f_height, f_width = filter_shape
    win_height, win_width = window

    pad_height, resize_height = reconstructParam(input_size=height,
                                                 filter_size=f_height,
                                                 window_size=win_height)
    pad_width, resize_width = reconstructParam(input_size=width,
                                               filter_size=f_width,
                                               window_size=win_width)

    if resize_height != height or resize_width != width:
        result = []

        for data in input_data:
            # cv2.resize 的 height, width 和一般的順序相反
            res = cv2.resize(data, (resize_width, resize_height))
            result.append(res)

        result = np.array(result)
    else:
        result = input_data.copy()

    if pad_height != 0 or pad_width != 0:
        result = np.pad(result, ((0, 0), (pad_height, pad_height), (pad_width, pad_width)))

    return result


# endregion


input_data = np.random.rand(1, 4, 4)
result = inputReconstruct(pad_ohlc, filter_shape=(4, 4), window=(3, 3))


def getCubicInterpolation(points, x):
    """
    cubic interpolation
    參考網站: https://www.paulinternet.nl/?page=bicubic

    f(x) = ax^3 + bx^2 + cx + d
    f'(x) = 3ax^2 + 2bx + c
    令 p0: x = -1, p1: x = 0, p2: x = 1, p3: x = 2

    f(0) = d = p1
    f(1) = a + b + c + d = p2
    f'(0) = c = (p2 - p0) / 2
    f'(1) = 3a + 2b + c = (p3 - p1) / 2

    a = (-0.5p0 + 1.5p1 - 1.5p2 + 0.5p3)
    b = (p0 - 2.5p1 + 2p2 - 0.5p3)
    c = (-0.5p0 + 0.5p2)
    d = p1

    f(p0, p1, p2, p3, x) = (-0.5p0 + 1.5p1 - 1.5p2 + 0.5p3)x^3 + (p0 - 2.5p1 + 2p2 - 0.5p3)x^2 + (-0.5p0 + 0.5p2)x + p1
    :param points:
    :param x:
    :return:
    """
    a = -0.5 * points[0] + 1.5 * points[1] - 1.5 * points[2] + 0.5 * points[3]
    b = points[0] - 2.5 * points[1] + 2 * points[2] - 0.5 * points[3]
    c = -0.5 * points[0] + 0.5 * points[2]
    d = points[1]

    return a * np.power(x, 3) + b * np.power(x, 2) + c * x + d


x = np.linspace(0, 15, 16)
y = np.sin(x)

x_pron = np.linspace(0, 15, 8)
y_pron = [y[0]]
n_pron = len(x_pron)

for i in range(1, n_pron - 1):
    index = x_pron[i]
    last_index = math.floor(index)
    next_index = math.ceil(index)
    before_last_index = max(last_index, 0)
    after_next_index = min(next_index, n_pron - 1)

    points = [y[before_last_index],
              y[last_index],
              y[next_index],
              y[after_next_index]]

    interpolate = getCubicInterpolation(points, index - last_index)
    y_pron.append(interpolate)

y_pron.append(y[-1])

plt.plot(x, y, "b-")
plt.plot(x_pron, y_pron, "r-")
plt.show()
