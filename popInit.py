import numpy as np


def popInitialization(NP, size):
    """
    初始化一个二维数组。

    参数：
    NP - - 数组的第一维度大小
    size - - 数组的第二维度大小

    返回值：
    xPopulation - - 初始化后的二维数组
    """
    # 使用numpy的zeros函数创建一个形状为(NP, size)的全零数组
    # 然后，对这个全零数组的每一个元素，加上一个[0,1)之间的随机数
    # 这样，我们就得到了一个形状为(NP, size)的数组，其中的每一个元素都是一个随机数
    xPopulation = np.zeros((NP, size)) + np.random.rand(NP, size)

    return xPopulation
