import loadData 

from kmeans_algorithm import kmeans
from seededKmeans_algorithm import seededKmeans
from fuzzyCmeans_algorithm import fuzzyCmeans
from EM_algorithm import expectation_maximization
from elm import elmMap

from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import math
import csv
mode = algo = 0
kernel = ''


# <------------------Choosing the Dataset and whether to use elm------------------------------>

dataFile2 = "20ng-all-stemmed.txt"
dataFile3 = "breast-cancer-wisconsin.data.txt"
dataFile1 = "r8-test-stemmed.txt"
dataFile4 = "movement_libras.data.txt"


# tfidf_matrix, label, name = loadData.datasetParkinson()
tfidf_matrix, label, name = loadData.libraDataset(dataFile4)
# tfidf_matrix, label, name = loadData.reuter8Dataset(dataFile1)
# tfidf_matrix, label, name = loadData.gestureDataset()
# tfidf_matrix, label, name = loadData._20ngDataset(dataFile2)
# tfidf_matrix, label, name = loadData.irisDataset()
# tfidf_matrix, label, name = loadData.datasetMnist()
# tfidf_matrix, label, name = loadData.breastCancerDataset(dataFile3)

no_of_clusters = len(set(label))
no_of_iterations = 10
fuzziness_coefficient = 2


#<<<---------ELM------------>>>>>
# tfidf_matrix = elmMap(tfidf_matrix, int(2*tfidf_matrix.shape[1]), 'sigmoid')
#print("elm result: \n", tfidf_matrix)
#print(elmMatrix.shape)




#<---------------------------------Opening the file to write----------------------------------->

iteration = list()
for i in range(0,no_of_iterations): 
	itr = "iteration" + str(i+1)
	iteration.append(itr)

score_file = open("resultsem.csv", 'a' )
scoreWriter = csv.writer(score_file, delimiter=",")


ans='y'


#<---------------------------------Choosing algorithm to run----------------------------------->

while(ans=='y'):
	print("\n1. kmeans \n2. seededKmeans \n3. fuzzyCmeans \n4. EM\n")
	algo = int(input("Choose algorithm to run: "))

	print("\n1. without kernel \n2. with kernel \n3. elm feature space \n")
	mode = int(input("Choose the mode of algo: "))

	if mode==1:
		modeOfRun = " Without Kernel "
	elif mode==2:
		modeOfRun = " With Kernel "
	elif mode==3:
		modeOfRun = " Using elm feature space "
	else:
		modeOfRun = "invalid"
		print("invalid mode")
		
	if mode==3:
		ratio = input("Enter the dimension of elm feature space : ")
		modeOfRun = modeOfRun + str(ratio)
		elm_ratio = float(ratio)*tfidf_matrix.shape[1]
		print(elm_ratio)
		
		print(modeOfRun)
		tfidf_matrix = elmMap(tfidf_matrix, elm_ratio, 'sigmoid')

	if mode==2:
		kernel = input("Enter the type of kernel (rbf, poly): ")
		modeOfRun = modeOfRun + kernel
		
	if algo==1:
		purity_scores, nmi_scores  = kmeans(tfidf_matrix, label, no_of_clusters, no_of_iterations)
		
		scoreWriter.writerow(["K MEANS"])
		scoreWriter.writerow([modeOfRun])
		scoreWriter.writerow([name])
		scoreWriter.writerow("")
		scoreWriter.writerow(["No of Docs:" ,tfidf_matrix.shape[0], "No. of Features:", tfidf_matrix.shape[1], "No. of Clusters", no_of_clusters, "No. of Iterations:", no_of_iterations])
		scoreWriter.writerow(iteration)
		scoreWriter.writerow(purity_scores)
		scoreWriter.writerow(nmi_scores)
		scoreWriter.writerow("")


	elif algo==2:
		purity_scores, nmi_scores  =  seededKmeans(tfidf_matrix, label, no_of_clusters, no_of_iterations)

		scoreWriter.writerow(["SEEDED K MEANS"])
		scoreWriter.writerow([modeOfRun])
		scoreWriter.writerow([name])
		scoreWriter.writerow("")
		scoreWriter.writerow(["No of Docs:" ,tfidf_matrix.shape[0], "No. of Features:", tfidf_matrix.shape[1], "No. of Clusters", no_of_clusters, "No. of Iterations:", no_of_iterations])
		scoreWriter.writerow(iteration)
		scoreWriter.writerow(purity_scores)
		scoreWriter.writerow(nmi_scores)
		scoreWriter.writerow("")


	elif algo==3:

		purity_scores, nmi_scores  = fuzzyCmeans(tfidf_matrix, label, no_of_clusters, fuzziness_coefficient, no_of_iterations)
		scoreWriter.writerow(["FUZZY C MEANS"])
		scoreWriter.writerow([modeOfRun])
		scoreWriter.writerow([name])
		scoreWriter.writerow("")
		scoreWriter.writerow(["No of Docs:" ,tfidf_matrix.shape[0], "No. of Features:", tfidf_matrix.shape[1], "No. of Clusters", no_of_clusters, "No. of Iterations:", no_of_iterations])
		scoreWriter.writerow(iteration)
		scoreWriter.writerow(purity_scores)
		scoreWriter.writerow(nmi_scores)
		scoreWriter.writerow("")


	elif algo==4:
		scoreWriter.writerow(["EXPECTATION MAXIMIZATION"])
		scoreWriter.writerow([modeOfRun])
		scoreWriter.writerow([name])
		scoreWriter.writerow("")
		purity_scores, nmi_scores  = expectation_maximization(tfidf_matrix, label, no_of_clusters, 1)
		scoreWriter.writerow(["No of Docs:" ,tfidf_matrix.shape[0], "No. of Features:", tfidf_matrix.shape[1], "No. of Clusters", no_of_clusters, "No. of Iterations:", no_of_iterations])
		scoreWriter.writerow(iteration)
		scoreWriter.writerow(purity_scores)
		scoreWriter.writerow(nmi_scores)
		scoreWriter.writerow("")

		
	else:
		print("\n Invalid algo code chosen.\n")
	
	scoreWriter.writerow("")
	scoreWriter.writerow("")

	ans =  input("do you want to run algorithms again on same dataset? (y/n): ")

#<---------------------------------Enter 'y' to run more tests on same dataset----------------------------------->

print("main over")
