
from numpy import *

def loadDataSet(fileName):      # function to parse delimited floats
    dataMat = []                # dataMat is a list
    fr = open(fileName)
    for line in fr.readlines():
        curLine = line.strip().split(',')
        fltLine = map(float,curLine)      # map all elements to float()
        dataMat.append(fltLine)
    return dataMat              # returns list of lists of data points

def distEclud(vecA, vecB):      # returns Euclidean distance between points
    return sqrt(sum(power(vecA - vecB, 2)))

def randCent(dataSet, k):         # create k random initial cluster centers
    n = shape(dataSet)[1]       # n = number of dimensions of input examples
    centroids = mat(zeros((k,n))) # create k x n centroid matrix
    randArr = random.choice(150,size=k,replace=False)
    print randArr
    for j in range(k):            # create random cluster centers (j = dimension)
                                  #   within bounds of each dimension
        centroids[j] = dataSet[(randArr[j])]
        #minJ = min(dataSet[:,j])
        #maxJ = max(dataSet[:,j])
        #rangeJ = float(maxJ - minJ)
        #centroids[:,j] = mat(minJ + rangeJ * random.rand(k,1))
    return centroids

def kMeans(dataSet,k,distMeas=distEclud,createCent=randCent):
    m,n = shape(dataSet)                # m = number of data points
    means = mean(dataSet, axis=0)
    stdvs = std(dataSet, axis=0)
    for i in range(n):
        for j in range(m):
            newVal = (((dataSet[j,i])-(means[0,i]))/(stdvs[0,i]))
            dataSet[j,i] = newVal
    clusterAssment = mat(zeros((m,2)))  # create matrix of data point assigns (col 0)
                                        #   col 0: index of point's assigned cluster
                                        #   col 1: point's distance^2 to center
    centroids = createCent(dataSet, k)  # create k initial random centroids
    print "initial centroids=\n",centroids
    clusterChanged = True               # run algorithm at least one iteration
    while clusterChanged:               # while cluster changes still occurring
        clusterChanged = False             # assume no changes occur (to terminate)
        # E step (assign data points to closest centroids)
        for i in range(m):                 # for each data point i
            minDist = inf; minIndex = -1
            for j in range(k):                       # for each centroid j
                distJI = distMeas(centroids[j,:],dataSet[i,:])  # dist between i, j
                if distJI < minDist:
                    minDist = distJI; minIndex = j       # assign i to closest centroid j
            if clusterAssment[i,0] != minIndex: clusterChanged = True
            clusterAssment[i,:] = minIndex,minDist**2    # save i's cluster assignment
        print "centroids =\n",centroids
        # M step (recalculate centroid locations)
        for c in range(k):                # for each centroid c
            ptsInClust = dataSet[nonzero(clusterAssment[:,0].A==c)[0]]
                                                       # get all points in cluster c
                                                       # self.A is np.asarray(self)
            centroids[c,:] = mean(ptsInClust, axis=0)  # assign centroid to mean
    print "final centroids =\n",centroids,"\n"
    #print "final assignments =\n",clusterAssment
    return centroids, clusterAssment

def getClassesInCluster(centroids,clusters,want):
    numSet=numVers=numVirg = 0
    for i in range(50):
        if (clusters[i,0]==want):
            numSet+=1
        if (clusters[(i+50),0]==want):
            numVers+=1
        if (clusters[(i+100),0]==want):
            numVirg+=1
    return numSet,numVers,numVirg

def printResults(centroids,clusters,fid):
    print clusters
    centroidRows,centroidCols = centroids.shape
    clusterRows,clusterCols = clusters.shape
    fid.write("iris Data, K = "+str(centroidRows)+"\n\n")
    for i in range(centroidRows):
        fid.write("cluster "+str(i)+": ")
        savetxt(fid,centroids[i,:],delimiter =",")
        if (i == (centroidRows-1)):
            fid.write("\n")

    total = 0
    for i in range(centroidRows):
        counter = 0
        cTotal = 0
        for j in range(clusterRows):
            if clusters[j,0] == i:
                counter += 1
                cTotal += clusters[j,1]
        cTotal = cTotal/counter
        total += cTotal
        fid.write("cluster "+str(i)+" size: "+str(counter)+
        ", mean dist^2: "+str(cTotal)+"\n")
        if (i == (centroidRows-1)):
            fid.write("\n")
    total = total/centroidRows
    fid.write("overall mean dist^2: "+str(total)+"\n\n")

    for i in range(centroidRows):
        tempSet,tempVers,tempVirg = getClassesInCluster(centroids,clusters,i)
        fid.write("cluster "+str(i)+" profile: "+str(tempSet)+" "
        +str(tempVers)+" "+str(tempVirg)+"\n")
        if (i == (centroidRows-1)):
            fid.write("\n")
