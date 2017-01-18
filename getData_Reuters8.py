from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
import numpy as np

labelMap = dict()
label = list()
#getting raw input from the file
def gestureDataset():
	path = os.getcwd() + "/gesture_phase_dataset/"
	os.chdir(path)

	files = ['a1','a2','a3','b1','b3','c1','c3']
	inputList = list()
	i = 0
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

	print(len(inputList))
	print(inputList[1])

	return inputList, label


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

	# print(docList[0])
	# print(inputList[0])

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
	tfidf_matrix = tfidf_vectorizer.fit_transform(docList)
	# no_of_docs = tfidf_matrix.shape[0]
	# no_of_terms = tfidf_matrix.shape[1]
	# print(no_of_docs, no_of_terms)
	#print("original: \n", tfidf_matrix)
	tfidf_matrix = normalize(tfidf_matrix, axis=0)
	#print("normalized : \n" ,tfidf_matrix)
	#print("tf_idf matrix shape : " , tfidf_matrix.shape)

	return tfidf_matrix, label



