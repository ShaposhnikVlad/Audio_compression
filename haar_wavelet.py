#!/usr/bin/env python3

input_file = open("main.wav", 'rb')

input_data = input_file.read(33554476)

data_arr = []
l_ch_data = []
r_ch_data = []

l_index = 0
r_index = 0

for i in range(33554432):
	data_arr.append( input_data[ i + 44 ] )
	if i%2 == 0:
		l_ch_data.append( data_arr[ i ] )
		
		l_index += 1
	else:
		r_ch_data.append( data_arr[ i ] )

		r_index += 1

for j in range(5):
	print('L', j + 1, ' : ', l_ch_data[j]) 
	print('R', j + 1, ' : ', r_ch_data[j])
