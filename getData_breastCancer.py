import numpy as np
# from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
label = list()
labelMap = [2,4]

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

	print(label)
	# np.array(inputList[)
	tfidf_matrix = np.matrix(inputList)
	
	tfidf_matrix = normalize(tfidf_matrix, axis=0)
	print(tfidf_matrix.shape)
	return tfidf_matrix, label, no_of_classes



#breastCancerDataset("breast-cancer-wisconsin.data.txt")
