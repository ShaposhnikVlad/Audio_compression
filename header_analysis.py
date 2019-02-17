#!/usr/bin/env python3

# opening file and reading data

#file_name = "crystallize.wav"

file_name = "new.wav"

input_file = open(file_name, 'rb')

print("File \"",file_name, "\" is opened!")

print("Info:") 

input_data = input_file.read(44)

#for i in range(44):
	#print( i, ": ", input_data[i] )


# find and print the number of channels

number_of_channels = input_data[23]

number_of_channels <<= 8 

number_of_channels |= input_data[22]

print("	Number of channels:", number_of_channels)


# find and print the sample rate

interval = range(27,23,-1)

sample_rate = 0

for i in interval:
	#print( input_data[i] )

	sample_rate <<= 8 

	sample_rate |= input_data[i]

print("	Sample rate:", sample_rate, "Hz")


# find and print the byte rate

interval = range(31,27,-1)

byte_rate = 0

for i in interval:
	#print( input_data[i] )

	byte_rate <<= 8 

	byte_rate |= input_data[i]

print("	Byte rate:", byte_rate, "byte/sec")


# find and print the block size

interval = range(33,31,-1)

block_size = 0

for i in interval:
	#print( input_data[i] )

	block_size <<= 8 

	block_size |= input_data[i]

print("	Block size:", block_size, "byte")


# find and print the number of bits per sample

interval = range(35,33,-1)

bits_per_sample = 0

for i in interval:
	#print( input_data[i] )

	bits_per_sample <<= 8 

	bits_per_sample |= input_data[i]

print("	Bits per sample:", bits_per_sample, "bits")


#find and print the size of data for all samples

interval = range(43,39,-1)

size_of_sample_data = 0

for i in interval:
	#print( input_data[i] )

	size_of_sample_data <<= 8 

	size_of_sample_data |= input_data[i]

print("	Size of the sample data:", size_of_sample_data, "byte")

input_file.close()


