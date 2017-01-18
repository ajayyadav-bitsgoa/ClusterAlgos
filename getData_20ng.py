from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
inputList = list()
labelMap = dict()
label = list()

def _20ngDataset(filename):
	rawInput = open(filename,'r')

	for line in rawInput:
	    inputList.append((line.strip()).split('\t'))


	
	for i in range(0, len(inputList)):
		if(len(inputList[i]) ==1):
			inputList[i].append('\0')

	
	#print(len(inputList[1073]))
	# for i in range(0, len(inputList)):
	# 	try:
	# 		print(inputList[i][1])
	# 	except:
	# 		print(" error at index : ", i)
	# 		break
	
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
	print(tfidf_matrix.shape[0])
	print(tfidf_matrix.shape[1])
	print(len(labelMap))
	# no_of_terms = tfidf_matrix.shape[1]
	# print(no_of_docs, no_of_terms)
	#print("original: \n", tfidf_matrix)
	tfidf_matrix = normalize(tfidf_matrix, axis=0)

	return tfidf_matrix, label

_20ngDataset("20ng-all-stemmed.txt")