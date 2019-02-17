#!/usr/bin/env python3

# opening file and reading data

file_name = "crystallize.wav"

input_file = open(file_name, 'rb')

print("File \"",file_name, "\" is opened!")

input_data = input_file.read(33554476)

#print(input_data)

output_file = open("main.wav", 'wb')

output_file.write(input_data);

output_file.close()

input_file.close()

input_file1 = open("new.wav", 'rb')

print("Info:") 

input_data1 = input_file1.read(7680044)

for i in range(44):
	print( i, ": ", input_data[i] )

