import kMeans as km
from numpy import *

fid = open("irisResults.txt","w")

datMat = mat(km.loadDataSet("irisData2.txt"));
centers,assigns = km.kMeans(datMat,8)
km.printResults(centers,assigns,fid)
