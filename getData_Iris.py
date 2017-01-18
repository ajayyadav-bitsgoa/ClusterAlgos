from sklearn import datasets
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
labelMap = [0,1,2]

def irisDataset():
	iris = datasets.load_iris()
	X = iris.data 
	Y = iris.target
	X = normalize(X, axis=0)
	return X, Y

#irisDataset()