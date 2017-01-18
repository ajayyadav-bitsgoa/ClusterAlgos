import numpy as np
import math
import matplotlib.pyplot as plt
import random
from sklearn.metrics.cluster import normalized_mutual_info_score
from kernel import distance



def fuzzyCmeans(tfidf__matrix, label_, no_of__clusters, fuzziness__coefficient, no_of_iterations):
	
	global tfidf_matrix, label, no_of_docs, no_of_terms , no_of_clusters, fuzziness_coefficient, total_purity, nmi, purity_scores, nmi_scores
	
	tfidf_matrix = tfidf__matrix
	label = label_
	no_of_docs = int(tfidf_matrix.shape[0])	
	no_of_terms = int(tfidf_matrix.shape[1])
	no_of_clusters = no_of__clusters
	fuzziness_coefficient = fuzziness__coefficient
	total_purity = 0
	nmi = 0
	purity_scores = list()
	nmi_scores = list()

	global membership_matrix, documentClusterDistribution, clusterPurity, centroid_vectors, clusterDist, clusterClass

	membership_matrix = np.zeros(shape=(tfidf_matrix.shape[0], no_of_clusters))
	centroid_vectors = np.zeros(shape=(no_of_clusters, tfidf_matrix.shape[1]))
	documentClusterDistribution = np.zeros(shape=(no_of_clusters, len(set(label))), dtype=int)
	clusterClass = np.zeros(shape=(no_of_docs,), dtype=int)
	clusterDist = np.zeros(shape=(no_of_clusters,))
	clusterPurity = np.zeros(shape=(no_of_clusters,))

	# print(tfidf_matrix.shape)
	print("\t\t\t FUZZY C MEANS \n\n------x--------x---------x---------x-----------x----------x-------\n\n")
	print("docs : ", no_of_docs, "features : ", no_of_terms, "clusters : ",no_of_clusters)

	#initialization of cluster centroids by randomly dividing the dataset
	tempCluster = np.zeros(no_of_terms)

	arr = np.arange(0, no_of_docs)
	random.seed(1)
	random.shuffle(arr)
	# print(arr)
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

	#algorithm
	iterations = 0
	while(iterations < no_of_iterations):

		print("\n................................................\n")
		print("iteration ", iterations+1)
		update_membership_matrix()	
		# print("Membership Matrix : " , membership_matrix)

		getDocumentClusterMatrix()
		print("Document Cluster Distribution :\n", documentClusterDistribution)
		
		purityFuzzyCmeans()
		print("purity : ", clusterPurity)

		overallPurity()
		print("total_purity : ", total_purity*100 , " %")

		nmi =  normalized_mutual_info_score(label, clusterClass)
		print("nmi score : ", nmi)

		update_centroids()
		# print("Centroid Vectors : \n",centroid_vectors)
		
		purity_scores.append(total_purity)
		nmi_scores.append(nmi)


		#plt.plot(clusterPurity)
		#plt.show()
		iterations += 1

	return purity_scores, nmi_scores

def getDocumentClusterMatrix():
	global documentClusterDistribution , label, clusterClass
	documentClusterDistribution.fill(0)

	for i in range(0, no_of_docs):
		documentClusterDistribution[np.argmax(membership_matrix[i])][label[i]] +=1
		clusterClass[i] = np.argmax(membership_matrix[i])

def update_membership_matrix():
		global clusterDist, membership_matrix 
		for i in range(0, no_of_docs):
			tempDist = 0
			for j in range(0, no_of_clusters):
				clusterDist[j] = getDistance(i,j, fuzziness_coefficient)
				tempDist += 1/clusterDist[j]
				
			for j in range(0, no_of_clusters):
				membership_matrix[i][j] = 1/(clusterDist[j]*tempDist)

def update_centroids():
	global centroid_vectors
	centroid_vectors.fill(0)   #error 2
	coeff = 0
	sum_coeff = 0
	for j in range(0, no_of_clusters):
		sum_coeff = 0
		for i in range(0, no_of_docs):
			coeff = math.pow(membership_matrix[i][j],fuzziness_coefficient)
			centroid_vectors[j] += tfidf_matrix[i]*coeff
			sum_coeff += coeff
		try:
			centroid_vectors[j] = centroid_vectors[j] / sum_coeff
		except:
			centroid_vectors[j] = 0

def getDistance(a,b, fuzziness_coefficient):
	
	d = centroid_vectors[b] - tfidf_matrix[a]
	d = math.pow(np.dot(d,d.T), 1/(fuzziness_coefficient-1))

	# d = distance(tfidf_matrix[i], centroid_vectors[j], kernel='poly')

	return d

def purityFuzzyCmeans():
	global clusterPurity
	for i in range(0, no_of_clusters):
		if np.sum(documentClusterDistribution[i]) != 0:
			clusterPurity[i] = np.max(documentClusterDistribution[i])/np.sum(documentClusterDistribution[i])
		else:
			clusterPurity[i] = 0

def overallPurity():
	global total_purity
	total_purity = 0
	for i in range(0, no_of_clusters):
		total_purity += clusterPurity[i]*np.sum(documentClusterDistribution[i])
	total_purity /= no_of_docs




