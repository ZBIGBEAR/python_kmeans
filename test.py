import kmeans
from numpy import *
import matplotlib.pyplot as plt
plt.switch_backend('agg')

if __name__=="__main__":
	count=int(input("please input points:"))
	k=int(input("please input k:"))
	#从文件中读取点集
	dataMat = mat(kmeans.loadDataSet('data/testSet.txt'))
	#随机生成n个点，横纵坐标范围是[i,j]
	#dataMat = kmeans.loadDataSetByRand(count,0,10)
	#print(dataMat1)
	#print("===================")
	#print(dataMat)
	#centroids = kmeans.randCent(dataMat,k)
	#print(centroids)
	#print(kmeans.distEclud(dataMat[0],dataMat[1]))
	#print("=============================================")
	centroids,clustAssing = kmeans.kmeans(dataMat,k)
	#print("myCentroids")
	#print(myCentroids)
	print("clustAssing")
	print(clustAssing)
	
	#将点集按照类别分类
	#classify = [[]]*k
	#classify = [[],[],[],[]]
	classify=[[] for i in range(k)]
	m = len(clustAssing)
	for i in range(m):
		classify[int(clustAssing[i].A[0][0])].append(list(dataMat[i].A[0]))
	#将点及其质心画出来
	fig = plt.figure()
	ax = fig.add_subplot(111)
	colors = ['r','g','y','b']
	for i in range(k):
		ax.scatter([point[0] for point in classify[i] ],[point[1] for point in classify[i]],s=10,c=colors[i])
		ax.scatter(centroids.A[i][0],centroids.A[i][1],s=30,c=colors[i],marker='x')
	#画质心
	#print(centroids.A[0])
	#ax.scatter([ point[0] for point in centroids.A ],[ point[1] for point in centroids.A ],s=10,c='red',marker='x')
	plt.savefig(str(k)+'.jpg')
