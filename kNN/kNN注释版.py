from numpy import *
import operator
from matplotlib import *
import matplotlib.pyplot as plt


def createDataSet():
	group = array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
	labels = ['A','A','B','B']
	return group, labels

def classify0(inX, dataSet, lables, k):
	'''
	inX是待分类的点,dataSet是带标签的点,标签存在lables中,k就是k近邻的k
	'''
	dataSetSize = dataSet.shape[0]
	# '''
	# 将inX(两个元素的点)拼成跟dataSet一样大小的矩阵,并对应相减
	# e.g.
	# tile([1,2],(2,2))->[[1,2,1,2],[1,2,1,2]]
	# '''
	diffMat = tile(inX, (dataSetSize, 1)) - dataSet
	# '''
	# [[1,2],[3,4]].sum(axis=1)->[3,7]
	# dataSet的形状如下:
	# x     y
	# x1    y1
	# x2    y2
	# ...   ...
	# '''
	sqDistances = (diffMat**2).sum(axis=1)
	distances = sqDistances**0.5
	#argsort()是排序后的下标array
	sortedDistIndicies = distances.argsort()
	classCount={}
	for i in range(k):
		voteIlabel = labels[sortedDistIndicies[i]]
		classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
	#这里完全可以不用排序,直接线性查找最大元素即可
	sortedClassCount = sorted(classCount.iteritems(),
	  key=operator.itemgetter(1), reverse=True)
	return sortedClassCount[0][0]

def file2matrix(filename):
	#读取数据'datingTestSet'函数
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

