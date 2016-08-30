# -*- coding:utf-8 -*-

from numpy import *


def sigmoid(intX):
    return 1.0 / (1 + exp(-intX))


def gradAscent(dataMatIn, classLabels):
    dataMatrix = mat(dataMatIn)
    labelMat = mat(classLabels).transpose()
    m,n=shape(dataMatrix)
    alpha = 0.001
    maxCycles = 500
    weights = ones((n,1))
    for k in xrange(maxCycles):
        h = sigmoid(dataMatrix*weights)
        error = labelMat-h
        weights=weights+alpha*dataMatrix.transpose()*error
    return weights
