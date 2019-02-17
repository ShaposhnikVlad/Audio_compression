#!/usr/bin/env python3
from math import sqrt

#---------------------------- FUNCTIONS DECLARATION

#---------------------------- Fill table
def fill_table(table, size, value):

	i_1 = 0
	i_2 = 1

	for i in range(int(size/2)):
		table[i][i_1] = value
		table[i][i_2] = value
		i_1 += 2
		i_2 += 2

	i_1 = 0
	i_2 = 1

	for i in range(int(size/2)):
		table[i+int(size/2)][i_1] = value
		table[i+int(size/2)][i_2] = -value
		i_1 += 2
		i_2 += 2
#----------------------------

#---------------------------- Matrix multiplication

def matr_mult(table, x, size):

	result = []

	index = 0

	for i in range(int(size/2)):
		tmp = 0

		for j in range(2):
			tmp = tmp + int(table[i][j + index] * x[j + index])

		result.append(int(tmp))

		index += 2


	index = 0

	for i in range(int(size/2)):
		tmp = 0

		for j in range(2):
			tmp = tmp + int(table[i + int(size/2)][j + index] * x[j + index])

		result.append(int(tmp))

		index += 2

	#print(result)
	
	return result

#---------------------------- 


#---------------------------- Wavelet transform

def wavelet_transform(x, size):

	value = 1/sqrt(2)

	a = []

	vect = []

	flag = 1

	while flag:

		table = [[0] * size for i in range(size)]

		fill_table(table, size, value)

		a = matr_mult(table, x, size)

		print('a: ',a)

		for i in range(int(size/2)):
			vect.append(a[size - i - 1])

		print(vect)

		x.clear()

		for i in range(int(size/2)):
			x.append(a[i])

		size = size >> 1

		if len(a) == 2:

			vect.append(a[0])

			flag = 0

	vect.reverse()

	print(vect)

	print('a0 :', a[0])

	return vect

#---------------------------- 


#---------------------------- END OF FUNCTION DECLARATION



#---------------------------- MAIN 

x = [13, 11, 15, 13, 21, 9, 24, 6]

size = 8

result_vector = []

result_vector = wavelet_transform(x,size) 

#---------------------------- END OF MAIN 




	
