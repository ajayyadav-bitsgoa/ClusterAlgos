# import math
# a = [1,2,3,4,5]
# b = [[1,2,3],[4,5,6]]




# def duc(ba):
# 	global a, b
# 	c = sum(math.pow(a[i],2) for i in range(0,5))
# 	print(c)
# 	# a = ba
# 	# a = [1213,121,121]
# 	#duc2()
# 	# a = [456,678,9090]

# def duc2():
# 	c = a 
# 	d = b
# 	print(c,d, "\t", a)

# duc(b)
# # print(a)
# # duc2()

import csv

csvfile = open("exp.csv", 'w' , newline='')

csvwriter = csv.writer(csvfile, delimiter=",")

csvwriter.writerow(["K-MEANS"])
csvwriter.writerow("")
iteration = list()
for i in range(0,10):
	itr = "iteration" + str(i)
	iteration.append(itr)

csvwriter.writerow(iteration)

