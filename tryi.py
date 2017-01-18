# def character(filename):
# 	rawInput = open(filename,'r')
# 	for line in rawInput:
#  		print(line)



# character("mixoutALL_shifted.mat")

import csv

csvfile = open("exp.csv", 'w' , newline='')

csvwriter = csv.writer(csvfile, delimiter=' ', quotechar='|')

csvwriter.writerow("Hi ajay! ")
