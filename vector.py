#!/usr/bin/env python3


def average(vector, size, res_vector):

	#cur_res_vector = []
	
	a_vector = []

	index = 0

	for i in range(int(size/2)):
		a_vector.append( int((vector[index] + vector[index+1]) / 2) )

		res_vector.append( int((vector[index] - vector[index+1]) / 2) )

		index += 2

	#print("------Step------")

	#print("a_vector: ", a_vector)

	#print("res_vector: ", res_vector)

	#if size == 1:
		#res_vector.append(a_vector[0])	
		
	return a_vector

def wavelet_transform(vector, size):
	
	current_result_vector = []

	tmp = []

	tmp = average(vector, size, current_result_vector)

	size = size / 2

	while size != 1:
		tmp = average( tmp, size, current_result_vector)
		
		size = int( size / 2 )

	current_result_vector.append(tmp[0])	

	last_a = current_result_vector.pop( len(current_result_vector) - 1 )

	#print(current_result_vector)

	#print(last_a)	

	current_result_vector.insert(0, last_a)
	
	#print(current_result_vector)

	return current_result_vector

"""def rev_vect(vector):

	print("a")

	return 1

	print("b")

	for i in range(1):
		print(i)

	#tmp = []

	#tmp.extend(vector)

	#tmp.reverse()	

	#return tmp
"""
#________________________________________________________________________________

# opening file and reading data

file_name = "new.wav"

input_file = open(file_name, 'rb')

print("File \"",file_name, "\" is opened!")

print("Info:") 

input_data = input_file.read(64)

for i in range(44):
	print( i, ": ", input_data[i] )

#sample_vector =  [13, 11, 15, 12, 21, 9, 13, 56]

print("Sample vector: ", input_data)

size = 8

result_vector = []

result_vector = wavelet_transform(input_data, size)

print("Result vector: ", result_vector)


