from kernel import distance

import matplotlib.pyplot as plt
import numpy as np
import math
import random
from sklearn.metrics.cluster import normalized_mutual_info_score


def kmeans(tfidf__matrix, _label, no_of__clusters, no_of_iterations):
	
	global tfidf_matrix, label, no_of_docs, no_of_terms , no_of_clusters, documentClusterDistribution, \
			clusterPurity, centroid_vectors, clusterDist, clusterClass, clusterCount, purity, nmi, purity_scores, nmi_scores

	tfidf_matrix = tfidf__matrix
	label = _label
	
	

	no_of_clusters = no_of__clusters
	no_of_docs = int(tfidf_matrix.shape[0])
	no_of_terms = int(tfidf_matrix.shape[1])
	iterations = 0
	purity = 0
	nmi = 0
	purity_scores = list()
	nmi_scores = list()
	
	print("\t\t\t K MEANS \n\n------x--------x---------x---------x-----------x----------x-------\n\n")
	print("docs : ", no_of_docs, "features : ", no_of_terms, "clusters : ",no_of_clusters)

	documentClusterDistribution = np.zeros(shape=(no_of_clusters, len(set(label))), dtype = int)
	clusterPurity = np.zeros(shape=(no_of_clusters,))

	#array to store cluster vectors
	centroid_vectors = np.zeros(shape=(no_of_clusters, no_of_terms))

	#array to store distance of a particular document from all clusters respectively
	clusterDist = np.zeros(shape=(no_of_clusters,))

	#array to store labels for documents based on distance from clusters
	clusterClass = np.zeros(shape=(no_of_docs,), dtype=int)
	
	#aray to store no of documents in each cluster
	clusterCount = np.zeros(shape=(no_of_clusters,), dtype=int)

	#temporary array to initial all cluster centroids
	tempCluster = np.zeros(no_of_terms)

	#initialization of cluster centroids by randomly dividing the dataset
	arr = np.arange(0, no_of_docs)
	random.seed(1)
	random.shuffle(arr)
	#print(arr)
	k = no_of_clusters
	p = int(tfidf_matrix.shape[0]/k)
	start = 0
	stop = p
	while(k):
	    if(k==1):
	        stop = no_of_docs
	    tempCluster.fill(0)       #error 1 
	    for i in range(start,stop):
	        tempCluster += tfidf_matrix[arr[i]] 
	    tempCluster = tempCluster/(stop-start)
	    #print(type(tempCluster))
	    centroid_vectors[no_of_clusters-k] = tempCluster
	    start = stop
	    stop = stop + p
	    k -= 1

	

	#the algorithm
	# epsilon = 0.001


	while(iterations < no_of_iterations):
		
		print("\n................................................\n")
		print("iteration ", iterations+1)

		assignClusters()
		purityKmeans()
		overallPurity()
		nmi = normalized_mutual_info_score(label, clusterClass)
		print("distribution: \n", documentClusterDistribution)
		print("clusterPurity :\n", clusterPurity)
		print("overall purity : ", purity*100, " % \n")
		print("nmi score : ", nmi )
		
		updateClusters()
		purity_scores.append(purity)
		nmi_scores.append(nmi)

		iterations += 1
	return purity_scores, nmi_scores

def assignClusters():
	
	global clusterCount, documentClusterDistribution, clusterClass

	clusterCount.fill(0)
	documentClusterDistribution.fill(0)
	
	
	for i in range(0, no_of_docs):
		for j in range(0, no_of_clusters):
			d = centroid_vectors[j] - tfidf_matrix[i]
			d = np.dot(d,d.T)
			
			# d = distance(tfidf_matrix[i], centroid_vectors[j], kernel='poly')

			clusterDist[j] = d
		clusterClass[i] = np.argmin(clusterDist)
		clusterCount[clusterClass[i]] += 1
		documentClusterDistribution[clusterClass[i]][label[i]]  += 1


def updateClusters():
	
	global centroid_vectors
	centroid_vectors.fill(0)
	
	for i in range(0, no_of_docs):
	    centroid_vectors[clusterClass[i]] += tfidf_matrix[i]
	
	for i in range(0,no_of_clusters):
		if clusterCount[i] != 0 :
			centroid_vectors[i] = centroid_vectors[i] / clusterCount[i]
		else :
			centroid_vectors[i] = 0



def purityKmeans():
	
	global clusterPurity
	for i in range(0, no_of_clusters):
		if np.sum(documentClusterDistribution[i]) != 0 :
			clusterPurity[i] = np.max(documentClusterDistribution[i])/np.sum(documentClusterDistribution[i])
		else :
			clusterPurity[i] = 0


def overallPurity():
	global purity
	purity = 0
	for i in range(0, no_of_clusters):
		purity += clusterPurity[i]*clusterCount[i]
	purity /= no_of_docs
	
