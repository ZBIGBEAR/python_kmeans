from numpy import *
#import random

#随机生成点集
def loadDataSetByRand(n,i,j):
	dataMat = []
	for x in range(n):
		tmp = []
		#print(random.uniform(i,j),i,j)
		tmp.append(float(random.uniform(i,j)))
		tmp.append(float(random.uniform(i,j)))
		dataMat.append(tmp)
		#dataMat.append([random.uniform(i,j),random.uniform(i,j)])
	#print(mat(dataMat))
	#print("-------------")
	return mat(dataMat)

#加载数据集
def loadDataSet(filename):
	dataMat = []
	fr = open(filename)
	for line in fr.readlines():
		curline = line.strip().split('\t')
		curArr = []
		for c in curline:
			curArr.append(float(c))
		dataMat.append(curArr)
	#print(shape(dataMat))
	return dataMat

#计算两个向量的欧式距离
def distEclud(vecA,vecB):
	return sqrt(sum(power(vecA-vecB,2)))

#根据数据集构造一个包含k个随机质心得集合,质心必须在数据集范围内
def randCent(dataSet,k):
	n = shape(dataSet)[1]
	centroids = mat(zeros((k,n)))
	for j in range(n):
		minJ = min(dataSet[:,j])
		rangeJ = float(max(dataSet[:,j])-minJ)
		#random.randn(n,m)表示生成一个n*m的矩阵
		centroids[:,j]=minJ+rangeJ*random.randn(k,1)
	return centroids

#kmeans算法
def kmeans(dataSet,k,distMeas=distEclud,createCent=randCent):
	#dataSet，传入得是一个80*2得矩阵
	#shape，获取矩阵的类型，dataSet的类型为(80,2)，表示80行2列，因此m=80
	m = shape(dataSet)[0]
	#print("m")
	#print(m)
	#print("clusterAssment")
	#zeros((m,2))表示获取一个m*2的0值数组。因此clusterAssment是一个80*2的、每个元素为0的矩阵
	clusterAssment = mat(zeros((m,2)))
	#print(clusterAssment)
	#k=4时，centroids是一个4*4的矩阵
	centroids = createCent(dataSet,k)
	#print("centroids")
	#print(centroids)
	clusterChanged = True
	while clusterChanged:
		#print("while===============================")
		clusterChanged = False
		for i in range(m):
			#遍历所有的待分类的点
			minDist = inf
			minIndex = -1
			for j in range(k):
				#遍历所有的质心，求出与当前分类点最近的那个质心
				distJI = distMeas(centroids[j,:],dataSet[i,:])
				if distJI < minDist:
					minDist = distJI
					minIndex = j
			#记录每个点所属的质心以及其与质心得距离
			if clusterAssment[i,0] != minIndex:
				clusterChanged = True
				#print("clusterChanged",clusterChanged)
				clusterAssment[i,:]=minIndex,minDist**2
		
		#print(centroids)
		#print("clusterAssment")
		#print(clusterAssment)
		#print("ptsInClust")
		#print(ptsInClust)
		#break
		#print("=========================================",clusterChanged)
		for cent in range(k):
			#print("nonzero(clusterAssment[:,0])")
			#print(clusterAssment[:,0])
			#print(clusterAssment[:,0].A)
			#print(clusterAssment[:,0].A==cent)
			#print(nonzero(clusterAssment[:,0].A==cent))
			#clusterAssment[:,0].A表示获取所有点的所属质心，是一个n*1的矩阵
			#clusterAssment[:,0].A==cent表示获取数组clusterAssment[:,0].A中值等于cent的元素，输出是一个n*1矩阵，等于则为True,不等于则为False
			#nonzero(clusterAssment[:,0].A==cent)。nonzero表示输出矩阵中不为0的元素的下标，这里输入的是n*1的矩阵，则nonzero输出的是x*2的矩阵，两列，第一列是不为0元素的行坐标，第二类是不为0元素的列坐标，这里列坐标肯定都是0。x表示不为0元素的数量
			#nonzero(clusterAssment[:,0].A==cent)[0]获取到的就是不为0元素的行坐标
			#ptsInClust获取到的是不为0元素的点集合
			ptsInClust = dataSet[nonzero(clusterAssment[:,0].A==cent)[0]]
			#print("ptsInClust")
			#print(ptsInClust)
			#break
			#mean(ptsInClust,axis=0)求这些点的各列的平均值，也就是横坐标的平均值和纵坐标的平均值
			centroids[cent,:]=mean(ptsInClust,axis=0)
			#print("centroids")
			#print(centroids)
			#print(mean(ptsInClust,axis=0))
		#print("centroids")
		#print(centroids)
		#break
	return centroids,clusterAssment
