from math import exp
import numpy as np
from random import normalvariate  # 正态分布
from datetime import datetime
import pandas as pd

#sigmoid函数
def sigmoid(inx):
    #return 1. / (1. + exp(-max(min(inx, 15.), -15.)))
    return 1.0 / (1 + exp(-inx))

def SGD_FM(dataMatrix, classLabels, k, iter):
    '''
    :param dataMatrix:  特征矩阵
    :param classLabels: 类别矩阵
    :param k:           辅助向量的大小
    :param iter:        迭代次数
    :return w：         一阶特征的系数wi
            v：         辅助向量，生成wij
            w_0：       w0
    '''
    # dataMatrix用的是mat, classLabels是列表
    m, n = np.shape(dataMatrix)   #矩阵的行列数，即样本数和特征数
    alpha = 0.01
    # 初始化参数
    # w = random.randn(n, 1)#其中n是特征的个数
    w = np.zeros((n, 1))      #一阶特征的系数
    w_0 = 0.
    v = normalvariate(0, 0.2) * np.ones((n, k))   #即生成辅助向量，用来训练二阶交叉特征的系数

    for it in range(iter):
        for x in range(m):  # 随机优化，每次只使用一个样本
            # 二阶项的计算
            inter_1 = dataMatrix[x] * v
            inter_2 = np.multiply(dataMatrix[x], dataMatrix[x]) * np.multiply(v, v)  #二阶交叉项的计算
            interaction = sum(np.multiply(inter_1, inter_1) - inter_2) / 2.       #二阶交叉项计算完成

            p = w_0 + dataMatrix[x] * w + interaction  # 计算预测的输出，即FM的全部项之和
            loss = 1-sigmoid(classLabels[x] * p[0, 0])    #计算损失

            # 随机梯度下降（SGD）训练参数
            w_0 = w_0 +alpha * loss * classLabels[x]
            for i in range(n):
                if dataMatrix[x, i] != 0:
                    w[i, 0] = w[i, 0] +alpha * loss * classLabels[x] * dataMatrix[x, i]
                    for j in range(k):
                        v[i, j] = v[i, j]+ alpha * loss * classLabels[x] * \
                                  (dataMatrix[x, i] * inter_1[0, j] - v[i, j] * dataMatrix[x, i] * dataMatrix[x, i])
        print("第{}次迭代后的损失为{}".format(it, loss))

    return w_0, w, v