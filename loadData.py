from sklearn import datasets
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
import numpy as np
import os, csv 

label = []
labelMap = {}

def datasetParkinson():
	X = np.load("Parkinson_X.npy")
	Y = np.load("Parkinson_Y.npy")
	X = normalize(X, axis=0)
	return X, Y, "Parkinson"


def datasetMnist():
	X = np.load("mnist_X.npy")
	Y = np.load("mnist_Y.npy")
	X = normalize(X, axis=0)
	return X, Y, "Minst"

def gestureDataset():
	global labelMap, label
	path = os.getcwd() + "/gesture_phase_dataset/"
	os.chdir(path)

	files = ['a1','a2','a3','b1','b3','c1','c3']
	inputList = list()
	# data = np.loadtxt("Data/sim.csv", delimiter=',', skiprows=1, usecols=range(1,15))
	c = 0
	for f in files:
		filename = f + '_raw.csv'
		csvfile = open(filename, 'r')

		csvreader = csv.reader(csvfile) #, delimiter=' ', quotechar = '|')

		next(csvreader, None) # skipping the first row
		for row in csvreader:
			one = []
			for r in row[0:18]:
				one.append(float(r))
			inputList.append(one)
			# print(row[19])
			if row[19] not in labelMap and len(labelMap) <= 5 :
				labelMap[row[19]] = c
				c += 1
				
			label.append(labelMap[row[19]])
		
	inputList = normalize(inputList, axis=0)


	return inputList, label, "Gesture Dataset"


def irisDataset():
	iris = datasets.load_iris()
	X = iris.data 
	Y = iris.target
	X = normalize(X, axis=0)
	return X, Y, "Iris Dataset"


def libraDataset(filename):
	data = open(filename,'r')
	inputList = []
	le = []
	for line in data:
		le = []
		for l in line.strip().split(','):
			#print(line)
				le.append(float(l))
		label.append(le[-1])

			# print(le)
		inputList.append(le[0:-1])

	# print(len(inputList[0]))
	for i in range(len(label)):
		label[i] -= 1
			
	
	tfidf_matrix = np.matrix(inputList)
	
	tfidf_matrix = normalize(tfidf_matrix, axis=0)
	print(tfidf_matrix.shape)
	print(set(label))
	return tfidf_matrix, label, "Libra Dataset"


def breastCancerDataset(filename):
	data = open(filename,'r')
	inputList = []
	le = []
	for line in data:
		le = []
		for l in line.strip().split(','):
			#print(line)
			if l == '?':
				le.append(0)
			else:
				le.append(int(l))
		label.append(le[-1])

			# print(le)
		inputList.append(le[1:-1])

	print(len(inputList[0]))
	for i in range(len(label)):
		if label[i] == 2:
			label[i] = 0
		else:
			label[i] = 1

	# print(label)
	# np.array(inputList[)
	tfidf_matrix = np.matrix(inputList)
	
	tfidf_matrix = normalize(tfidf_matrix, axis=0)
	# print(tfidf_matrix.shape)
	return tfidf_matrix, label, "Breast Cancer Dataset"

def reuter8Dataset(filename): 
	rawInput = open(filename,'r')
	inputList = list()

	#Converting the labelled dataset and storing into a list
	# Using REUTER 25178 R8 Dataset
	for line in rawInput:
	    inputList.append((line.strip()).split('\t'))

	#Removing the label of the documents
	docList = list()
	for i in range(0, len(inputList)):
	    docList.append(inputList[i][1])

	
	#mapping the labels to numbers
	
	c = 0
	for i in range(0,len(inputList)):
		if inputList[i][0] not in labelMap:
			labelMap[inputList[i][0]] = c
			c = c + 1

	for i in range(0,len(inputList)):
		label.append(labelMap[inputList[i][0]])
	
	#Converting the documents into vector form (term-document matrix)
	tfidf_vectorizer = TfidfVectorizer(min_df = 1)
	tfidf_matrix = tfidf_vectorizer.fit_transform(docList[0:1000])
	# no_of_docs = tfidf_matrix.shape[0]
	# no_of_terms = tfidf_matrix.shape[1]
	# print(no_of_docs, no_of_terms)
	#print("original: \n", tfidf_matrix)
	tfidf_matrix = normalize(tfidf_matrix, axis=0)
	#print("normalized : \n" ,tfidf_matrix)
	print("tf_idf matrix shape : " , tfidf_matrix.shape)

	return tfidf_matrix, label[0:1000], "Reuter 8 Dataset"





def _20ngDataset(filename):
	rawInput = open(filename,'r')
	inputList =[]
	
	for line in rawInput:
	    inputList.append((line.strip()).split('\t'))


	
	for i in range(0, len(inputList)):
		if(len(inputList[i]) ==1):
			inputList[i].append('\0')

	
	
	
	docList = list()
	for i in range(0, len(inputList)):
	    docList.append(inputList[i][1])

	c = 0
	for i in range(0,len(inputList)):
		if inputList[i][0] not in labelMap:
			labelMap[inputList[i][0]] = c
			c = c + 1

	for i in range(0,len(inputList)):
		label.append(labelMap[inputList[i][0]])



	tfidf_vectorizer = TfidfVectorizer(min_df = 1)
	tfidf_matrix = tfidf_vectorizer.fit_transform(docList)
	
	tfidf_matrix = normalize(tfidf_matrix, axis=0)

	return tfidf_matrix, label, "20ng Dataset"


