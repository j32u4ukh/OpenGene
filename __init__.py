from abc import ABCMeta, abstractmethod

"""
TODO: 不要局限於學術上對基因、基因組、細胞等的定義，根據自己需求來定義應該更適合
bit (位元) ， 電腦最小單位
Byte (位元組) ， 1Byte = 8 bits
KB ( Kilobyte )       = 1024 Byte
MB ( Megabyte )       = 1024 KB
GB ( Gigabyte )       = 1024 MB
TB ( Terabyte )       = 1024 GB
PB ( Petabyte )       = 1024 TB
EB ( Exabyte )        = 1024 PB

1 Kbit = 1024 bits
"""


class Kbit(metaclass=ABCMeta):
    def __init__(self):
        """ 1 Kbit = 1024 bit
        每個 Kbit 由 1024 個 bit (位元，1 或 0) 所組成，24 bits 定義 Kbit 種類(16777216 種)，1000 bits 定義數值內容。
        OpenGene 中最小物件或結構，細胞。
        """
        pass


class Mbit:
    def __init__(self):
        """ 1 Mbit = 1024 Kbit
        處理特定任務中的子項目，例如視覺任務中的圖片分類，貓狗分類應該可以，但要對每個像素分類，進而到語意分類或許有些困難？
        由類似功能的"細胞"所組成的"功能區"。
        """
        pass

"""
以上，隨著訓練會不斷切換 Kbit 的組成，不同的 Kbit 所需 bit 數量也不同，因此無法將目前多餘的部分移除，因為種類改變後可能會需要。
以下，雖然數值部分可能隨著學習而做調整，細胞間的連結也可能更新，但不再需要冗餘的 bit，可將多餘的部分排除，進而提升效率。
"""


class Gbit:
    def __init__(self):
        """ 1 Gbit = 1024 Mbit
        處理特定任務的綜合項目，例如可同時處理視覺任務中的圖片分類、語意分類、GAN 等。
        由多個類似的"功能區"所組成的"組織"。
        """
        pass


class Tbit:
    def __init__(self):
        """ 1 Tbit = 1024 Gbit
        "個體"的基本單位，偕同多個"組織"來達成複合型任務，例如利用視覺追蹤特定目標，再利用運動系統控制四肢去抓取物品。
        "個體"可透過刻意訓練，變成"專家"。
        """
        pass


class Pbit:
    def __init__(self):
        """ 1 Pbit = 1024 Tbit
        由多個專長類似的"個體"組成的"專家組織"，可從不同觀點來解決更困難的任務。
        """
        pass


class Ebit:
    def __init__(self):
        """ 1 Pbit = 1024 Tbit
        由多個"專家組織"之間的競爭，"專家"可在各個組織之間重新組合成不同的組織。
        """
        pass
