import numpy as np
from sklearn.preprocessing import normalize
import os, csv 


def gestureDataset():
	path = os.getcwd() + "/gesture_phase_dataset/"
	os.chdir(path)

	files = ['a1','a2','a3','b1','b3','c1','c3']
	inputList = []
	labelMap = {}
	label =[]
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

gestureDataset()