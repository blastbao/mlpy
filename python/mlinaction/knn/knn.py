from numpy import *
import operator
from tqdm import tqdm


def createDataset():
    group = array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels


def classify0(inX, dataset, labels, k):
    datasetSize = dataset.shape[0]
    diffMat = tile(inX, (datasetSize, 1)) - dataset
    sqDiffMat = diffMat**2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances**0.5
    sortedDistIndicies = distances.argsort()
    classCount = {}
    for i in range(k):
        voteILabel = labels[sortedDistIndicies[i]]
        classCount[voteILabel] = classCount.get(voteILabel, 0) + 1
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]


def file2matrix(filename):
    fr = open(filename)
    arrayOLines = fr.readlines()
    numberOfLines = len(arrayOLines)
    returnMat = zeros((numberOfLines, 3))
    classLabelVector = []
    index = 0
    for line in arrayOLines:
        line = line.strip()
        listFromLine = line.split('\t')
        returnMat[index, :] = listFromLine[0:3]
        classLabelVector.append(int(listFromLine[-1]))
        index += 1
    return returnMat, classLabelVector


def auto_norm(dataset):
    minValues = dataset.min(0)
    maxValues = dataset.max(0)
    ranges = maxValues - minValues
    normDataset = zeros(shape(dataset))
    m = dataset.shape[0]
    normDataset = dataset - tile(minValues, (m, 1))
    normDataset = normDataset/tile(ranges, (m, 1))
    return normDataset, ranges, minValues


def datingClassTest():
    hoRatio = 0.10
    datingDataMat, datingLabels = file2matrix("datingTestSet.txt")
    normMat, ranges, minValues = auto_norm(datingDataMat)
    m = normMat.shape[0]
    numTestVecs = int(m*hoRatio)
    errCount = 0.0
    for i in tqdm(range(numTestVecs)):
        classifierResult = classify0(normMat[i, :], normMat[numTestVecs:m, :], datingLabels[numTestVecs:m], 3)
        print("the classifier came back with: {}, the real answer is: {}".format(classifierResult, datingLabels[i]))
        if classifierResult != datingLabels[i]:
            errCount += 1.0
    print("the total error rate is: {}%".format(errCount/float(numTestVecs)))
