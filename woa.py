# 导入必要的库
import math
import random
import time

import numpy
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import KFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from ReliefF import reliefFScore, top_select
from popInit import popInitialization


# 输入为数据集，最小最大标准化、relief F进行降维，输出为数据和标签
def processData(dataset):
    rawData = np.loadtxt(dataset, delimiter=',',
                         encoding='utf-8-sig')
    X = rawData[:, :-1]
    Y = rawData[:, -1]

    # 对数据进行min-MAX标准化
    minMAX = preprocessing.MinMaxScaler()
    X = minMAX.fit_transform(X)

    # 使用filter进行初步降维
    if X.shape[1] >= 7000:
        Score = reliefFScore(X, Y)
        top_index = top_select(Score)
        X = X[:, top_index]
    return X, Y


# 寻找最优解：错误率最低的
def findBest(pop, fit):
    sort = fit.argsort()
    return sort[0]


# Parameters
lb = 0
ub = 1
thres = 0.5
b = 1  # constant
maxGen = 100
NP = 80


def countFeatures(X):
    num = 0
    for d in range(size):
        if X[0][d] >= 0.5:
            num = num + 1
    return num


# Objective function:错误率
def callFitness(pop):
    popNum = pop.shape[0]  # 获取个体数量
    tempFitness = np.zeros(popNum)  # 存储当前种群所有个体的适应度值

    for i in range(popNum):
        xIndex = []
        for j in range(size):
            if pop[i][j] >= 0.5:
                xIndex.append(j)
        # kNN报错：特征数降为0了
        if len(xIndex) == 0:
            xTest = feat
        else:
            xTest = feat[:, xIndex]
        clf = KNeighborsClassifier(n_neighbors=5, algorithm='auto', metric='manhattan', n_jobs=-1)
        kf = KFold(n_splits=5, shuffle=True)
        tempFitness[i] = 1 - cross_val_score(clf, xTest, label, cv=kf, n_jobs=-1).mean()

    return tempFitness  # 返回错误率

if __name__ == "__main__":
    # Load Datasets
    feat, label = processData("D:\\OneDrive\\datasets\\de\\nci9.csv")

    # Number of dimensions
    size = feat.shape[1]

    # Initial
    gen = 0
    pop = popInitialization(NP=NP, size=size)
    fit = np.zeros((NP, 1))
    fit = callFitness(pop)

    xBest = np.ones((1, size))
    xBestFit = 1

    currentBestIndex = findBest(pop, fit)
    if xBestFit > fit[currentBestIndex]:
        xBestFit = fit[currentBestIndex]
        xBest = pop[currentBestIndex].reshape(1, -1)

    yVal = np.zeros(maxGen)

    while gen < maxGen:
        a = 2 - gen * (2 / maxGen)
        for i in range(NP):
            A = 2 * a * random.random() - a
            C = 2 * random.random()
            p = random.random()
            l = -1 + 2 * random.random()
            # Whale position update
            if p < 0.5:
                # Encircling prey
                if abs(A) < 1:
                    for d in range(size):
                        Dx = abs(C * xBest[0][d] - pop[i][d])
                        pop[i][d] = xBest[0][d] - A * Dx
                # Search for prey
                elif abs(A) >= 1:
                    for d in range(size):
                        k = random.randint(0, NP - 1)
                        Dx = abs(C * pop[k][d] - pop[i][d])
                        pop[i][d] = pop[k][d] - A * Dx
            # Bubble-net attacking
            elif p >= 0.5:
                for d in range(size):
                    dist = abs(xBest[0][d] - pop[i][d])
                    pop[i][d] = dist * math.exp(b * l) * math.cos(2 * numpy.pi * l) + xBest[0][d]
            # Boundary
            pop[i] = np.clip(pop[i], 0, 1).reshape(1, -1)

        fit = callFitness(pop)
        currentBestIndex = findBest(pop, fit)
        if xBestFit > fit[currentBestIndex]:
            xBestFit = fit[currentBestIndex]
            xBest = pop[currentBestIndex].reshape(1, -1)
        yVal[gen] = (1 - xBestFit) * 100
        print("第", gen + 1, "次准确率为：", yVal[gen])
        gen = gen + 1

    # 最后输出图像
    x = np.arange(0, maxGen)
    y = np.array(yVal)
    # plt.scatter(x,y)
    plt.ylim((0, 100))
    plt.plot(x, y)
    plt.show()
