"""
TODO: 結構一開始提供直線型串接(LinearStructure)，最終應提供網狀結構，'直線型'為'網狀結構'的退化型。

TODO: 網狀結構類別，用於管理細胞結構的管理，某些結構可能造成無窮迴圈(或許容許無窮迴圈的結構存在，但利用每次傳遞訊號強度都會衰退，
讓訊號的傳遞可以自然的傳遞或衰退)
"""
from abc import ABCMeta, abstractmethod


class Structure(metaclass=ABCMeta):
    def __init__(self, cells):
        # cells 根據是"直線型"還是"網狀結構"，將傳入 list 和 graph，因此這裡不標註變數類型
        self.cells = cells

    @abstractmethod
    def parseStructure(self, matrix):
        pass
