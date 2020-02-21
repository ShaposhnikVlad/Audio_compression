import heapq
import os
from functools import total_ordering
from struct import *

@total_ordering
class HeapNode:

    def __init__(self, val, freq):
        self.value = val
        self.frequence = freq
        self.left = None
        self.right = None

    def __lt__(self, other):

        return self.frequence < other.frequence

    def __eq__(self, other):
        if(other == None):
            return False
        if(not isinstance(other, HeapNode)):
            return False
        return self.frequence == other.frequence


class HuffmanCoding:

    def __init__(self, seq):

        self.sequence = seq
        self.heap = []
        self.codes = {}
        self.reverse_mapping = {}

    def crt_freq_dict(self, seq):
        freq_dict = {}

        for unit in seq:
            if not unit in freq_dict:
                freq_dict[unit] = 0

            freq_dict[unit] += 1

        #print("Freq dict:", freq_dict)

        return freq_dict



    def crt_heap(self, freq_dict):
        for key in freq_dict:
            node = HeapNode(key, freq_dict[key])
            heapq.heappush(self.heap, node)



    def merge_nodes(self):
        while(len(self.heap) > 1):
            node_1 = heapq.heappop(self.heap)
            node_2 = heapq.heappop(self.heap)

            merged_nodes = HeapNode(None, node_1.frequence + node_2.frequence)
            merged_nodes.left = node_1
            merged_nodes.right = node_2

            heapq.heappush(self.heap, merged_nodes)



    def crt_codes(self):
        root = heapq.heappop(self.heap)
        cur_code = ""

        self.crt_codes_helper(root, cur_code)



    def crt_codes_helper(self, root, cur_code):
        if(root == None):
            return

        if(root.value != None):
            self.codes[root.value] = cur_code
            self.reverse_mapping[cur_code] = root.value
            return

        self.crt_codes_helper(root.left, cur_code + "0")
        self.crt_codes_helper(root.right, cur_code + "1")



    def crt_encoding(self, seq):
        encoding = ""

        for unit in seq:
            encoding += self.codes[unit]

        return encoding



    def pad_encoding(self, encoding):
        padding_len = 8 - len(encoding) % 8

        for i in range(padding_len):
            encoding += "0"

        padding_info = "{0:08b}".format(padding_len)
        encoding = padding_info + encoding

        return encoding



    def get_byte_seq(self, padded_encoding):
        if(len(padded_encoding) % 8 != 0):
            print("WRONG PADDING")
            exit(0)

        #byte_seq = bytearray()
        b = []

        for i in range(0, len(padded_encoding), 8):
            byte = padded_encoding[i:i+8]
            #byte_seq.append(int(byte,2))
            b.append(int(byte,2))


        return b



    def compress(self, chan_name):
        #out_name = chan_name + "_huffman_encoding.bin"

        #output_file = open(out_name, 'wb')

        freq = self.crt_freq_dict(self.sequence)
        self.crt_heap(freq)
        self.merge_nodes()
        self.crt_codes()

        #print(self.codes)

        encoding = self.crt_encoding(self.sequence)
        padded_encoding = self.pad_encoding(encoding)
        byte_seq = self.get_byte_seq(padded_encoding)

        #output_file.write(bytes(byte_seq))
        #output_file.write(b''.join(map(lambda x: pack("<B", x), byte_seq)))


        #output_file.close()

        #print("compressed")
        #return out_name

        return byte_seq


    # FUNCTIONS FOR DECODING

    def remove_padding(self, padded_encoding):
        padding_info = padded_encoding[:8]
        padding_len = int(padding_info, 2)

        padded_encoding = padded_encoding[8:]
        encoding = padded_encoding[:-1*padding_len]

        return encoding



    def decode_seq(self, encoding):
        cur_code = ""
        plain_seq = []

        for bit in encoding:
            cur_code += bit

            if cur_code in self.reverse_mapping:
                plain_seq.append(self.reverse_mapping[cur_code])
                cur_code = ""

        return plain_seq



    def decompress(self, huff_code):

        #plain_sequence = []
        bit_string = ""

        index = 0

        #byte = huff_code[index]

        #while len(byte) > 0:
        while index < len(huff_code):
            byte = huff_code[index]
            #bits = bin(byte)[2:].rjust(8,'0')
            bits = bin(byte)[2:].rjust(8, '0')
            bit_string += bits
            #byte = code.read(1)
            #print("Bit string", bit_string, type(bit_string))

            encoding = self.remove_padding(bit_string)

            decompressing = self.decode_seq(encoding)

            index += 1

        #plain_sequence.append(decompressing)

        #print("DECOMPRESSED")

        #print(decompressing)
        return decompressing








