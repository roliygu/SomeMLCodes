from numpy import *
import operator
from matplotlib import *
import matplotlib.pyplot as plt

def createDataSet():
	group = array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
	labels = ['A','A','B','B']
	return group, labels

def getMax(dic):
	val=-1
	for key in dic:
		if dic[key]>val:
			label = key
	return label
 
def classify0(inX, dataSet, lables, k):
	dataSetSize = dataSet.shape[0]
	diffMat = tile(inX, (dataSetSize, 1)) - dataSet
	sqDistances = (diffMat**2).sum(axis=1)
	distances = sqDistances**0.5
	sortedDistIndicies = distances.argsort()
	classCount={}
	for i in range(k):
		voteIlabel = lables[sortedDistIndicies[i]]
		classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
	return getMax(classCount)

def file2matrix(filename):
	fr = open(filename)
	arrayOLines = fr.readlines()
	numberOfLines = len(arrayOLines)
	returnMat = zeros((numberOfLines,3))
	classLabelVector = []
	index = 0
	for line in arrayOLines:
		line = line.strip()
		listFromLine = line.split('\t')
		returnMat[index,:] = listFromLine[0:3]
		classLabelVector.append(int(listFromLine[-1]))
		index += 1
	return returnMat,classLabelVector

def show(datingDataMat, datingLabels):
	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.scatter(datingDataMat[:,1], datingDataMat[:,0],
			15.0*array(datingLabels), 15.0*array(datingLabels))
	plt.show()

def autoNorm(dataSet):
	minVals = dataSet.min(0)
	maxVals = dataSet.max(0)
	ranges = maxVals - minVals
	normDataSet = zeros(shape(dataSet))
	m = dataSet.shape[0]
	normDataSet = dataSet - tile(minVals, (m,1))
	normDataSet = normDataSet/tile(ranges, (m,1))
	return normDataSet, ranges, minVals

def datingClassTest(): 
    datingDataMat,datingLabels = file2matrix('datingTestSet.txt')   
    normMat, ranges, minVals = autoNorm(datingDataMat)
    numTestVecs = normMat.shape[0]
    errorCount = 0.0
    for i in range(numTestVecs):
        classifierResult = classify0(normMat[i,:],normMat[:numTestVecs,:],\
        	datingLabels[:numTestVecs],4)
        if (classifierResult != datingLabels[i]): errorCount += 1.0
    print "the total error rate is: %f" % (errorCount/float(numTestVecs))
    print errorCount

def classifyPerson(percentTats, ffMiles, iceCream):
	datingDataMat, datingLabels = file2matrix('datingTestSet.txt')
	normMat, ranges, minVals = autoNorm(datingDataMat)
	inArr = array([percentTats, ffMiles, iceCream])
	classifierResult = classify0((inArr-minVals)/ranges,normMat,datingLabels,5)
	print classifierResult