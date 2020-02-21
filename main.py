from struct import *
from random import randint
from math import log
import os
import sys
import matplotlib.pyplot as plot
import matplotlib.ticker as ticker
from math import sqrt
import numpy as np
from pickle import *


def calc_block_size(n):  # must be 2^n size. Function gets n and calculate the 2^n size of block.
    return pow(2, int(n))


def val_count(wavelet_result):  # calculate count for all value in list

    wavelet_result.sort()

    size = len(wavelet_result)

    res = []

    index = 0

    member = ''

    while index < size:
        c = wavelet_result.count(wavelet_result[index])

        member = str(wavelet_result[index]) + ' : ' + str(c)

        res.append(member)

        index = index + c

    print(wavelet_result)

    for i in range(len(res)):
        print(res[i])


def sample_sequence_forming(byte_vector):  # No separating channels

    i = 0

    sample_seq = []

    while i < len(byte_vector):
        tmp = byte_vector[i + 1]

        tmp <<= 8

        tmp |= byte_vector[i]

        sample_seq.append(tmp)

        i += 2

    # print("sample_seq compliete")

    return sample_seq


def sample_sequence_forming_sep(byte_vector):  # Separating channels

    left_sample_seq.clear()

    right_sample_seq.clear()

    i = 0

    index = 0

    while i < len(byte_vector):

        if index & 1:

            tmp = byte_vector[i + 1]

            tmp <<= 8

            tmp |= byte_vector[i]

            right_sample_seq.append(tmp)

            index += 1

        else:
            tmp = byte_vector[i + 1]

            tmp <<= 8

            tmp |= byte_vector[i]

            left_sample_seq.append(tmp)

            index += 1

        i += 2


def signed_sample_forming(byte_vector, chn_count, byte_per_sample):
    # print("Try to be right this time.")

    if chn_count > 8:
        print("Wrong count of channels!")
        return 0

    if chn_count == 1:
        print("Not done yet for 1 channel.")

    elif chn_count == 2:

        sb_len = len(byte_vector) // (chn_count * byte_per_sample)  # sample block length

        # print(sb_len)

        l_chan = []
        r_chan = []

        s_index = 0  # current sample first byte index

        for i in range(sb_len):
            l_chan.append(unpack('<h', byte_vector[s_index: s_index + 2])[0])
            r_chan.append(unpack('<h', byte_vector[s_index + 2: s_index + 4])[0])

            s_index += 4

        return l_chan, r_chan

    elif chn_count == 3:
        print("Not done yet for 3 channels.")

    elif chn_count == 4:
        print("Not done yet for 4 channels.")

    elif chn_count == 5:
        print("Not done yet for 5 channels.")

    elif chn_count == 6:
        print("Not done yet for 6 channels.")

    elif chn_count == 7:
        print("Not done yet for 7 channels.")

    else:
        print("Not done yet for 8 channels.")


def bytes_to_sample(byte_seq, bits_per_sample):  # Universal byte2sample sequence forming function
    l_chan = []
    r_chan = []

    bytes_per_sample = bits_per_sample >> 3
    print("Bytes per sample:", bytes_per_sample)

    if bytes_per_sample == 1:
        struct_param = "<B"
    elif bytes_per_sample == 2:
        struct_param = "<H"
    elif bytes_per_sample == 4:
        struct_param = "<I"
    else:
        print("GOT SOME PROBLEMS HERE")
        sys.exit()

    count = len(byte_seq) // bytes_per_sample // 2

    j = 0

    for i in range(count):
        l_chan.append(unpack(struct_param, byte_seq[j:j + bytes_per_sample])[0])
        r_chan.append(unpack(struct_param, byte_seq[j + bytes_per_sample:j + (bytes_per_sample << 1)])[0])

        j += bytes_per_sample << 1

    return l_chan, r_chan


def byte_seq_forming(sample_vector):
    byte_seq = bytes(0)

    for i in range(len(sample_vector)):
        byte_seq += sample_vector[i].to_bytes(2, byteorder='little')

        print(i)

    # print(byte_seq)

    return byte_seq


def fast_byte_forming(sample_vector):
    byte_seq = ''

    byte_seq = b''.join(map(lambda x: pack("<H", x), sample_vector))

    print(byte_seq)


def stereo_decorrelation(l_chan, r_chan):
    diff_seq = []
    aver_seq = []

    for i in range(len(l_chan)):
        diff_seq.append(l_chan[i] - r_chan[i])
        aver_seq.append(l_chan[i] - (diff_seq[i] >> 1))

    # print("Diff: ", diff_seq)
    # print("Aver: ", aver_seq)

    return diff_seq, aver_seq


def inverse_decorrelation(diff_seq, aver_seq):
    l_chan = []
    r_chan = []

    for i in range(len(diff_seq)):
        if diff_seq[i] & 1 == 0:
            r_chan.append(aver_seq[i] - (diff_seq[i] >> 1))
            l_chan.append(diff_seq[i] + r_chan[i])
        else:
            r_chan.append(aver_seq[i] - (diff_seq[i] + 1 >> 1))
            l_chan.append(diff_seq[i] + r_chan[i])

    # print("L chan: ", l_chan)
    # print("R chan: ", r_chan)

    return l_chan, r_chan


def average_back(aver_amount, aver_diff, lost_flag):
    cur_x = []

    if lost_flag == "1":
        cur_x.append(aver_amount + aver_diff + 1)
        cur_x.append(aver_amount - aver_diff)
    else:
        cur_x.append(aver_amount + aver_diff)
        cur_x.append(aver_amount - aver_diff)

    return cur_x


def signed_average_back(a_vector, d_vector, lost_flag, detail_level):
    # print("*" * 20)

    # print("a: ", a_vector)
    # print("d: ", d_vector)
    # print("flag: ", lost_flag)
    # print("n: ", detail_level)

    hd_a_vector = []  # higher detailed vector

    for i in range(detail_level):

        if lost_flag.count(i):
            hd_a_vector.append(a_vector[i] + d_vector[i] + 1)
            hd_a_vector.append(a_vector[i] - d_vector[i])
        else:
            hd_a_vector.append(a_vector[i] + d_vector[i])
            hd_a_vector.append(a_vector[i] - d_vector[i])

    # print("Result: ", hd_a_vector)

    return hd_a_vector


def signed_inverse_wavelet_transform(wavelet_vector, lost_flag, n):
    current_a_vector = wavelet_vector[:1]

    for i in range(n):
        current_len = len(current_a_vector)

        current_d_vector = wavelet_vector[current_len: (current_len << 1)]

        current_lost_flag = lost_flag[-1 - i]

        current_a_vector = signed_average_back(current_a_vector, current_d_vector, current_lost_flag, current_len)

    return current_a_vector


def average(vector, size, res_vector, lost_flag):
    a_vector = []

    tmp_lost = ""

    index = size - 1

    half_size = size >> 1

    for i in range(half_size):
        sum = vector[index - 1] + vector[index]
        diff = vector[index - 1] - vector[index]

        if sum & 1 == 0:
            tmp_lost = "0" + tmp_lost
        else:
            tmp_lost = "1" + tmp_lost

        a_vector.insert(0, sum >> 1)
        # a_vector = [sum >> 1] + a_vector

        res_vector.insert(0, diff >> 1)

        index -= 2

    lost_flag = tmp_lost + lost_flag

    return a_vector, lost_flag


def signed_average(s_block, detail_level):  # sblock - sample block, sb_size - sample block size
    a_vector = []
    d_vector = []
    flag = []

    sb_size = calc_block_size(detail_level)

    pair_index = 1

    while pair_index <= sb_size:
        current_sum = s_block[pair_index - 1] + s_block[pair_index]
        current_diff = s_block[pair_index - 1] - s_block[pair_index]

        if current_sum & 1 == 1:  # save information lost positions
            flag.append((pair_index - 1) >> 1)

        a_vector.append(current_sum >> 1)
        d_vector.append(current_diff >> 1)

        pair_index += 2

        # if flag:
        # flag.insert(0, detail_level)

    return a_vector, d_vector, flag


def wavelet_transform(vector, size):
    current_result_vector = []

    lost_flag = ""

    tmp = vector

    while size != 1:
        tmp, lost_flag = average(tmp, size, current_result_vector, lost_flag)

        size = size >> 1

    # current_result_vector.append(tmp[0])

    # current_result_vector.insert(0, current_result_vector.pop(len(current_result_vector) - 1))
    current_result_vector = tmp[:1] + current_result_vector

    return current_result_vector, lost_flag


def signed_wavelet_transform(sblock, detail_level):
    wavelet_result = []

    flag = []

    current_a_vector = sblock

    # z = 0

    while detail_level > 0:
        # print("Step: ", z, ". Curr_a_vector: " , current_a_vector, ". Curr_sb_size: ", sb_size)

        current_a_vector, current_d_vector, current_flag = signed_average(current_a_vector, detail_level)

        wavelet_result = current_d_vector + wavelet_result

        # сравнить по скорости верхнее с этим:
        # current_d_vector.extend(wavelet_result)
        # wavelet_result = current_d_vector

        flag.append(current_flag)

        detail_level -= 1

        # z += 1

    # wavelet_result = current_a_vector + wavelet_result
    wavelet_result.insert(0, current_a_vector[0])

    # print("1: ", wavelet_result)

    return wavelet_result, flag


def insert_err(seq, err_size):
    for i in range(len(seq)):
        if seq[i] > (err_size * (-1)) and seq[i] < err_size:
            seq[i] = 0

    return seq


def inverse_wavelet_transform(vector, size, lost_flag):
    d_vector = []

    a_vector = []

    d_vector.extend(vector)

    a_vector.append(d_vector.pop(0))

    a_index = 0

    d_index = 0

    while d_index < len(d_vector):

        cur_x = average_back(a_vector.pop(a_index), d_vector[d_index], lost_flag[d_index])

        if cur_x[0] > 65535:
            a_vector.insert(a_index, 65535)
        else:
            a_vector.insert(a_index, abs(cur_x[0]))

        if cur_x[1] > 65535:
            a_vector.insert(a_index + 1, 65535)
        else:
            a_vector.insert(a_index + 1, abs(cur_x[1]))

        if a_index == len(a_vector) - 2:
            a_index = 0
        else:
            a_index = a_index + 2

        d_index = d_index + 1

    return a_vector


def last_sample_proc(last_sample_block):
    n = len(last_sample_block)

    filler_val = n

    # print("Old n: ", n)

    if (n & 1) == 1:
        last_sample_block.append(last_sample_block[n - 1])

        n += 1

    deg = 0

    new_n = n

    # print("New n: ", new_n)

    while n > 0:
        n >>= 1

        deg += 1

    deg -= 1

    # print("Degree: ", deg)

    if new_n == (2 << deg - 1):

        deg -= 1

        last_block_size = 2 << deg

        # print("Power of two. Last block size: ", last_block_size)

        # elif block_size == new_n:

        # last_block_size = block_size

        # print("Become full block. Last block size: ", last_block_size)

        # return 0, 0

    else:
        last_block_size = 2 << deg

        # print("Common case. Last block size: ", last_block_size)

    filler_val = last_block_size - filler_val

    # print("Filler value: ", filler_val)


    while last_block_size > new_n:
        last_sample_block.append(filler_val)

        new_n += 1

    # print(len(last_sample_block))

    return last_sample_block, deg + 1


def filler_val_cutter(last_sample_block):
    filler_size = last_sample_block[len(last_sample_block) - 1]

    # print("Filler size: ", filler_size)

    for i in range(filler_size):
        last_sample_block.pop()

    return last_sample_block


def coop_chan(left_chan, right_chan):  # cooperating channels

    size = len(left_chan)

    res = []

    for i in range(size):
        res.append(abs(left_chan[i]))
        res.append(abs(right_chan[i]))

    return res


def sep_compression(iter_count, block_size):  # Seems to work
    from Huffman import HuffmanCoding

    for i in range(iter_count):
        #print(i)

        input_data = input_file.read(block_size)

        sample_sequence_forming_sep(input_data)

        size = len(left_sample_seq)

        l_res, l_flag = wavelet_transform(left_sample_seq, size)
        r_res, r_flag = wavelet_transform(right_sample_seq, size)

        # l_res, r_res = stereo_decorrelation(left_sample_seq, right_sample_seq)

        # l_err = insert_err(l_res, 5)
        # r_err = insert_err(r_res, 5)

        # l_rle = rle(l_res, 2, 2)
        # r_rle = rle(r_res, 2, 2)

        # l_dec = rle_decoder(l_rle, 2)
        # r_dec = rle_decoder(r_rle, 2)

        # if (l_res != l_dec) or (r_res != r_dec):
        # print("User INVALID")
        # break

        l_huff = HuffmanCoding(l_res)
        r_huff = HuffmanCoding(r_res)

        l_code = l_huff.compress("L")
        r_code = r_huff.compress("R")

        # l_almost = l_huff.decompress(l_code)
        # r_almost = r_huff.decompress(r_code)

        # l_plain = inverse_wavelet_transform(l_almost, size, l_flag)
        # r_plain = inverse_wavelet_transform(r_almost, size, r_flag)

        # for i in range(len(l_plain)):
        # if l_plain[i] > 65535:
        #  print(l_plain[i], i)
        #   break

        # l_err = error_estimation(left_sample_seq, l_plain)
        # r_err = error_estimation(right_sample_seq, r_plain)
        # big_err.extend(l_err)
        # big_err.extend(r_err)

        # almost_plain = coop_chan(l_plain, r_plain)
        # code = coop_chan(l_code, r_code)

        # output_file.write(b''.join(map(lambda x: pack("<H", x), almost_plain)))
        left_code.write(b''.join(map(lambda x: pack("<B", x), l_code)))
        right_code.write(b''.join(map(lambda x: pack("<B", x), r_code)))

        # print("[ Complete:", int((i + 1) / iter_count * 100), "% ]", end='')

        # print('\r', end='')

    # last sample block (sep) processing

    print("Last i")

    input_data = input_file.read(block_size)

    sample_sequence_forming_sep(input_data)

    left_last_block, l_deg = last_sample_proc(left_sample_seq)
    right_last_block, l_deg = last_sample_proc(right_sample_seq)

    size = len(left_last_block)

    # l_res, r_res = stereo_decorrelation(left_last_block, right_last_block)

    l_res, l_flag = wavelet_transform(left_last_block, size)
    r_res, r_flag = wavelet_transform(right_last_block, size)

    # l_rle = rle(l_res, 2, 1)
    # r_rle = rle(r_res, 2, 1)

    # l_dec = rle_decoder(l_rle, 1)
    # r_dec = rle_decoder(r_rle, 1)

    # if (l_res != l_dec) or (r_res != r_dec):
    #   print("User INVALID")

    l_huff = HuffmanCoding(l_res)
    r_huff = HuffmanCoding(r_res)

    l_code = l_huff.compress("l")
    r_code = r_huff.compress("r")

    # l_almost = l_huff.decompress(l_code)
    # r_almost = r_huff.decompress(r_code)

    # l_plain = inverse_wavelet_transform(l_almost, size, l_flag)
    # r_plain = inverse_wavelet_transform(r_almost, size, r_flag)

    # cut_l = filler_val_cutter(l_plain)
    # cut_r = filler_val_cutter(r_plain)

    # l_err = error_estimation(l_seq, cut_l)
    # r_err = error_estimation(r_seq, cut_r)
    # big_err.extend(l_err)
    # big_err.extend(r_err)


    # almost_plain = coop_chan(cut_l, cut_r)

    left_code.write(b''.join(map(lambda x: pack("<B", x), l_code)))
    right_code.write(b''.join(map(lambda x: pack("<B", x), r_code)))

    # output_file.write(b''.join(map(lambda x: pack("<H", x), almost_plain)))


def basic_scheme(bb_size, iter_count, detail_level, predictor_type, pw_size):
    from Huffman import HuffmanCoding

    chn_count = 2
    byte_per_sample = 2

    for i in range(iter_count):

        in_data = input_file.read(bb_size)

        l_chan, r_chan = signed_sample_forming(in_data, chn_count, byte_per_sample)

        diff, aver = stereo_decorrelation(l_chan, r_chan)

        if predictor_type == 'rms':
            diff_predicted = rms_predictor(diff, pw_size)
            aver_predicted = rms_predictor(aver, pw_size)

        elif predictor_type == 'rmc':
            diff_predicted = rmc_predictor(diff, pw_size)
            aver_predicted = rmc_predictor(aver, pw_size)

        elif predictor_type == 'amp':
            diff_predicted = amp(diff, pw_size)
            aver_predicted = amp(aver, pw_size)

        else:
            print("Error! Unknown predictor type!")
            break

        diff_err = signed_calculate_error(diff, diff_predicted)
        aver_err = signed_calculate_error(aver, aver_predicted)

        diff_huff = HuffmanCoding(diff_err)
        aver_huff = HuffmanCoding(aver_err)

        diff_code = diff_huff.compress("DIFF")
        aver_code = aver_huff.compress("AVER")

        difference_code.write(b''.join(map(lambda x: pack("<B", x), diff_code)))
        average_code.write(b''.join(map(lambda x: pack("<B", x), aver_code)))

    print("Last block processing...")

    in_data = input_file.read(bb_size)

    l_chan, r_chan = signed_sample_forming(in_data, chn_count, byte_per_sample)

    left_lb, l_deg = last_sample_proc(l_chan)
    right_lb, r_deg = last_sample_proc(r_chan)

    diff_lb, aver_lb = stereo_decorrelation(left_lb, right_lb)

    if predictor_type == 'rms':
        diff_predicted = rms_predictor(diff_lb, pw_size)
        aver_predicted = rms_predictor(aver_lb, pw_size)

    elif predictor_type == 'rmc':
        diff_predicted = rmc_predictor(diff_lb, pw_size)
        aver_predicted = rmc_predictor(aver_lb, pw_size)

    elif predictor_type == 'amp':
        diff_predicted = amp(diff_lb, pw_size)
        aver_predicted = amp(aver_lb, pw_size)

    else:
        print("Error! Unknown predictor type!")

    diff_err = signed_calculate_error(diff_lb, diff_predicted)
    aver_err = signed_calculate_error(aver_lb, aver_predicted)

    diff_huff = HuffmanCoding(diff_err)
    aver_huff = HuffmanCoding(aver_err)

    diff_code = diff_huff.compress("DIFF")
    aver_code = aver_huff.compress("AVER")

    difference_code.write(b''.join(map(lambda x: pack("<B", x), diff_code)))
    average_code.write(b''.join(map(lambda x: pack("<B", x), aver_code)))


def compression_scheme_1(bb_size, iter_count, detail_level, predictor_type, pw_size):
    from Huffman import HuffmanCoding

    chn_count = 2
    byte_per_sample = 2

    for i in range(iter_count):
        in_data = input_file.read(bb_size)

        l_chan, r_chan = signed_sample_forming(in_data, chn_count, byte_per_sample)

        if predictor_type == 'rms':
            l_predicted = rms_predictor(l_chan, pw_size)
            r_predicted = rms_predictor(r_chan, pw_size)

        elif predictor_type == 'rmc':
            l_predicted = rmc_predictor(l_chan, pw_size)
            r_predicted = rmc_predictor(r_chan, pw_size)

        elif predictor_type == 'amp':
            l_predicted = amp(l_chan, pw_size)
            r_predicted = amp(r_chan, pw_size)

        elif predictor_type == 'no':
            l_predicted = []
            r_predicted = []

            l_predicted.extend(l_chan)
            r_predicted.extend(r_chan)

        else:
            print("Error! Unknown predictor type!")
            break

        l_err = signed_calculate_error(l_chan, l_predicted)
        r_err = signed_calculate_error(r_chan, r_predicted)

        #l_wavelet, l_flag = signed_wavelet_transform(l_err, detail_level - 2)
        #r_wavelet, r_flag = signed_wavelet_transform(r_err, detail_level - 2)

        #l_huff = HuffmanCoding(l_wavelet)
        #r_huff = HuffmanCoding(r_wavelet)

        diff, aver = stereo_decorrelation(l_err, r_err)

        l_huff = HuffmanCoding(diff)
        r_huff = HuffmanCoding(aver)

        l_code = l_huff.compress("DIFF")
        r_code = r_huff.compress("AVER")

        difference_code.write(b''.join(map(lambda x: pack("<B", x), l_code)))
        average_code.write(b''.join(map(lambda x: pack("<B", x), r_code)))


    print("Last block processing...")

    in_data = input_file.read(bb_size)

    l_chan, r_chan = signed_sample_forming(in_data, chn_count, byte_per_sample)

    left_lb, l_deg = last_sample_proc(l_chan)
    right_lb, r_deg = last_sample_proc(r_chan)

    if predictor_type == 'rms':
        l_predicted = rms_predictor(left_lb, pw_size)
        r_predicted = rms_predictor(right_lb, pw_size)

    elif predictor_type == 'rmc':
        l_predicted = rmc_predictor(left_lb, pw_size)
        r_predicted = rmc_predictor(right_lb, pw_size)

    elif predictor_type == 'amp':
        l_predicted = amp(left_lb, pw_size)
        r_predicted = amp(right_lb, pw_size)

    elif predictor_type == 'no':
        l_predicted = []
        r_predicted = []

        l_predicted.extend(l_chan)
        r_predicted.extend(r_chan)

    else:
        print("Error! Unknown predictor type!")

    l_err = signed_calculate_error(left_lb, l_predicted)
    r_err = signed_calculate_error(right_lb, r_predicted)

    #l_wavelet, l_flag = signed_wavelet_transform(l_err, l_deg - 2)
    #r_wavelet, r_flag = signed_wavelet_transform(r_err, r_deg - 2)

    #l_huff = HuffmanCoding(l_wavelet)
    #r_huff = HuffmanCoding(r_wavelet)

    diff, aver = stereo_decorrelation(l_err, r_err)

    l_huff = HuffmanCoding(diff)
    r_huff = HuffmanCoding(aver)

    l_code = l_huff.compress("DIFF")
    r_code = r_huff.compress("AVER")

    difference_code.write(b''.join(map(lambda x: pack("<B", x), l_code)))
    average_code.write(b''.join(map(lambda x: pack("<B", x), r_code)))


def compression_scheme_3(bb_size, iter_count, detail_level, p_type):
    from Huffman import HuffmanCoding

    chn_count = 2
    byte_per_sample = 2

    for i in range(iter_count):
        in_data = input_file.read(bb_size)

        l_chan, r_chan = signed_sample_forming(in_data, chn_count, byte_per_sample)

        if p_type == "wavelet":

            l_wavelet, l_flag = signed_wavelet_transform(l_chan, detail_level - 2)
            r_wavelet, r_flag = signed_wavelet_transform(r_chan, detail_level - 2)

            l_huff = HuffmanCoding(l_wavelet)
            r_huff = HuffmanCoding(r_wavelet)

            l_code = l_huff.compress("L")
            r_code = r_huff.compress("R")

            left_code.write(b''.join(map(lambda x: pack("<B", x), l_code)))
            right_code.write(b''.join(map(lambda x: pack("<B", x), r_code)))

        elif p_type == "decorrelation":

            diff, aver = stereo_decorrelation(l_chan, r_chan)

            l_huff = HuffmanCoding(diff)
            r_huff = HuffmanCoding(aver)

            l_code = l_huff.compress("DIFF")
            r_code = r_huff.compress("AVER")

            difference_code.write(b''.join(map(lambda x: pack("<B", x), l_code)))
            average_code.write(b''.join(map(lambda x: pack("<B", x), r_code)))

        else:
            print("MAIN P_TYPE ERROR")


    print("Last block processing...")

    in_data = input_file.read(bb_size)

    l_chan, r_chan = signed_sample_forming(in_data, chn_count, byte_per_sample)

    left_lb, l_deg = last_sample_proc(l_chan)
    right_lb, r_deg = last_sample_proc(r_chan)

    if p_type == "wavelet":

        l_wavelet, l_flag = signed_wavelet_transform(left_lb, l_deg - 2)
        r_wavelet, r_flag = signed_wavelet_transform(right_lb, r_deg - 2)

        l_huff = HuffmanCoding(l_wavelet)
        r_huff = HuffmanCoding(r_wavelet)

        l_code = l_huff.compress("L")
        r_code = r_huff.compress("R")

        left_code.write(b''.join(map(lambda x: pack("<B", x), l_code)))
        right_code.write(b''.join(map(lambda x: pack("<B", x), r_code)))

    elif p_type == "decorrelation":

        diff, aver = stereo_decorrelation(left_lb, right_lb)

        l_huff = HuffmanCoding(diff)
        r_huff = HuffmanCoding(aver)

        l_code = l_huff.compress("DIFF")
        r_code = r_huff.compress("AVER")

        difference_code.write(b''.join(map(lambda x: pack("<B", x), l_code)))
        average_code.write(b''.join(map(lambda x: pack("<B", x), r_code)))

    else:
        print("LAST BLOCK P_TYPE ERROR")


def signed_sep_decompression(bb_size, iter_count, detail_level, predictor_type, pw_size):
    from Huffman import HuffmanCoding

    chn_count = 2
    byte_per_sample = 2

    for i in range(iter_count):

        l_comp_file = open("/home/wb/PycharmProjects/course_work/HUFF_L_CHAN.bin", 'wb')
        r_comp_file = open("/home/wb/PycharmProjects/course_work/HUFF_R_CHAN.bin", 'wb')

        l_huff = l_comp_file.read()


def signed_sep_compression(bb_size, iter_count, detail_level, predictor_type, pw_size):
    from Huffman import HuffmanCoding

    chn_count = 2
    byte_per_sample = 2

    for i in range(iter_count):
        #print(i)

        in_data = input_file.read(bb_size)

        l_chan, r_chan = signed_sample_forming(in_data, chn_count, byte_per_sample)

        #pw_size = 3

        if predictor_type == 'rms':
            l_predicted = rms_predictor(l_chan, pw_size)
            r_predicted = rms_predictor(r_chan, pw_size)

        elif predictor_type == 'rmc':
            l_predicted = rmc_predictor(l_chan, pw_size)
            r_predicted = rmc_predictor(r_chan, pw_size)

        elif predictor_type == 'amp':
            l_predicted = amp(l_chan, pw_size)
            r_predicted = amp(r_chan, pw_size)

        elif predictor_type == 'no':
            l_predicted = []
            r_predicted = []

            l_predicted.extend(l_chan)
            r_predicted.extend(r_chan)

        else:
            print("Error! Unknown predictor type!")
            break

        l_err = signed_calculate_error(l_chan, l_predicted)
        r_err = signed_calculate_error(r_chan, r_predicted)

        l_wavelet, l_flag = signed_wavelet_transform(l_err, detail_level - 2)
        r_wavelet, r_flag = signed_wavelet_transform(r_err, detail_level - 2)

        l_huff = HuffmanCoding(l_wavelet)
        r_huff = HuffmanCoding(r_wavelet)

        l_code = l_huff.compress("L")
        r_code = r_huff.compress("R")

        #print(len(l_code))
        #print(len(r_code))

        #l_rice = rice_coder(l_wavelet)
        #r_rice = rice_coder(r_wavelet)

        left_code.write(b''.join(map(lambda x: pack("<B", x), l_code)))
        right_code.write(b''.join(map(lambda x: pack("<B", x), r_code)))

        #left_rice.write(b''.join(map(lambda x: pack("<B", x), l_rice)))
        #right_rice.write(b''.join(map(lambda x: pack("<B", x), r_rice)))

    print("Last block processing...")

    in_data = input_file.read(bb_size)

    l_chan, r_chan = signed_sample_forming(in_data, chn_count, byte_per_sample)

    left_lb, l_deg = last_sample_proc(l_chan)
    right_lb, r_deg = last_sample_proc(r_chan)

    #pw_size = 3

    if predictor_type == 'rms':
        l_predicted = rms_predictor(left_lb, pw_size)
        r_predicted = rms_predictor(right_lb, pw_size)

    elif predictor_type == 'rmc':
        l_predicted = rmc_predictor(left_lb, pw_size)
        r_predicted = rmc_predictor(right_lb, pw_size)

    elif predictor_type == 'amp':
        l_predicted = amp(left_lb, pw_size)
        r_predicted = amp(right_lb, pw_size)

    elif predictor_type == 'no':
        l_predicted = []
        r_predicted = []

        l_predicted.extend(l_chan)
        r_predicted.extend(r_chan)

    else:
        print("Error! Unknown predictor type!")

    l_err = signed_calculate_error(left_lb, l_predicted)
    r_err = signed_calculate_error(right_lb, r_predicted)

    l_wavelet, l_flag = signed_wavelet_transform(l_err, l_deg - 2)
    r_wavelet, r_flag = signed_wavelet_transform(r_err, r_deg - 2)

    l_huff = HuffmanCoding(l_wavelet)
    r_huff = HuffmanCoding(r_wavelet)

    l_code = l_huff.compress("L")
    r_code = r_huff.compress("R")

    #l_rice = rice_coder(l_wavelet)
    #r_rice = rice_coder(r_wavelet)

    left_code.write(b''.join(map(lambda x: pack("<B", x), l_code)))
    right_code.write(b''.join(map(lambda x: pack("<B", x), r_code)))

    #left_rice.write(b''.join(map(lambda x: pack("<B", x), l_rice)))
    #right_rice.write(b''.join(map(lambda x: pack("<B", x), r_rice)))


def no_sep_compression(iter_count, block_size):
    for i in range(iter_count):
        input_data = input_file.read(block_size)

        sample_vector = sample_sequence_forming(input_data)

        size = len(sample_vector)

        res = wavelet_transform(sample_vector, size)

        almost_plain = inverse_wavelet_transform(res, size)

        output_file.write(b''.join(map(lambda x: pack("<H", x), almost_plain)))

        print("[ Complete:", int((i + 1) / iter_count * 100), "% ]", end='')

        print('\r', end='')

    # last block (no sep) processing

    input_data = input_file.read(block_size)

    sample_vector = sample_sequence_forming(input_data)

    last_sample_vector = last_sample_proc(sample_vector, block_size)

    size = len(last_sample_vector)

    res = wavelet_transform(last_sample_vector, size)

    almost_plain = inverse_wavelet_transform(res, size)

    cutted_almost_plain = filler_val_cutter(almost_plain)

    output_file.write(b''.join(map(lambda x: pack("<H", x), cutted_almost_plain)))


def test_compression():
    from Huffman import HuffmanCoding

    in_file = open("/home/wb/Документы/course_work/crystallize.wav", 'rb')

    in_header = in_file.read(44)

    in_data = in_file.read(2 ** 16)

    sample_sequence_forming_sep(in_data)

    # print("L: ", left_sample_seq)
    # print("R: ", right_sample_seq)

    print("Plain seq len: ", len(left_sample_seq) << 1)

    size = len(left_sample_seq)

    # diff_chan, aver_chan = stereo_decorrelation(left_sample_seq, right_sample_seq)

    # l_chan, r_chan = inverse_decorrelation(diff_chan, aver_chan)

    # if left_sample_seq != l_chan or right_sample_seq != r_chan:
    # print("ERROR")

    l_res, l_flag = wavelet_transform(left_sample_seq, size)
    r_res, r_flag = wavelet_transform(right_sample_seq, size)

    # print("W(L): ", l_res)
    # print("W(R): ", r_res)

    # print()

    l_err = insert_err(l_res, 5)
    r_err = insert_err(r_res, 5)

    # print(l_err)

    # l_signs = signs_writer(l_res)
    # r_signs = signs_writer(r_res)

    # l_positive = list_abs(l_res)
    # r_positive = list_abs(r_res)

    # print(l_positive)

    # print("Plain len:", len(l_positive))

    # l_code = rice_coder(l_positive)
    # r_code = rice_coder(right_sample_seq)
    # print(l_code)

    # print("Encode len: ", len(l_code)//8)

    # l_decode = rice_decoder(l_code)
    # r_decode = rice_decoder(r_code)

    # print(l_decode)

    # if (l_decode != l_positive):
    #   print("BAD RICE")
    # else:
    #   print("GOOD RICE")

    # l_plain = signs_reader(l_positive, l_signs)
    # r_plain = signs_reader(r_positive, r_signs)

    # if (l_res != l_plain) or (r_res != r_plain):
    #     print("INVALID MASK")
    ## else:
    #  print("GOOD MASK")

    # print("W(L) length: ", len(l_res))
    # print("W(R) length: ", len(r_res))

    # l_rle = rle(l_res, 2, 1)
    # r_rle = rle(r_res, 2, 1)

    # print(l_rle)

    # print("RLE code len: ", len(l_err) + len(r_err))

    l_huff = HuffmanCoding(l_res)
    r_huff = HuffmanCoding(r_res)

    l_code = l_huff.compress("L")
    r_code = r_huff.compress("R")

    # print(l_code)
    # print(r_code)

    # lr_code = coop_chan(l_code, r_code)

    print("Code len: ", len(l_code) + len(r_code))



    # code_file.write(b''.join(map(lambda x: pack("<B", x), lr_code)))

    # print("Compressed file left name: ", l_name)
    # print("Compressed file right name: ", r_name)

    # l_decomp = l_huff.decompress(l_code)
    # r_decomp = r_huff.decompress(r_code)

    # print("Decode: ", l_decomp)

    # if l_decomp == l_res and r_decomp == r_res:
    #   print("GOOD HUFFMAN")
    # else:
    #  print("BAD HUFFMAN")

    # print("Left RLE length: ", len(l_rle))
    # print("Right RLE length: ", len(r_rle))

    # print(rr)

    # l_dec = rle_decoder(l_rle, 1)
    # r_dec = rle_decoder(r_rle, 1)

    # if (l_res != l_dec or r_res != r_dec):
    # print("BAD RLE")


    # l_almost = inverse_wavelet_transform(l_err, size, l_flag)
    # r_almost = inverse_wavelet_transform(r_res, size, r_flag)

    # if left_sample_seq != l_almost:
    # print("BAD W")
    # else:
    # print("Good W")
    # print("Almost: ", l_almost)

    # error_estimation(left_sample_seq, l_almost)


def error_estimation(plain_seq, result_seq):
    err_values = []

    null_count = 0

    for i in range(len(plain_seq)):

        if (plain_seq[i] == 0) or (result_seq[i] == 0):
            null_count += 1
        else:
            b = plain_seq[i] / result_seq[i]

            if (20 * log(b, 10)) > 1:
                err_values.append((plain_seq[i], result_seq[i]))

    # print("Error values: ", err_values)
    # print("Null count: ", null_count)
    print("Error count: ", len(err_values))


def list_abs(list):
    positive_list = []

    for i in range(len(list)):
        if list[i] < 0:
            positive_list.append(abs(list[i]))
        else:
            positive_list.append(list[i])

    return positive_list


def signs_writer(sequence):
    # writes 0 if number is positive, 1 if number is negative.

    signs_value = 0  # first number in wavelet_result will always be the positive number

    for i in range(len(sequence)):
        signs_value <<= 1

        if sequence[i] < 0:
            signs_value |= 1
        else:
            signs_value |= 0

    # print(bin(signs_value))
    return signs_value


def signs_reader(list, signs_value):
    print("SIGNS READER")

    plain_seq = []

    value_len = len(bin(signs_value)) - 2

    diff = len(list) - value_len

    current_bit = 1 << (value_len - 1)

    index = 0

    while index < diff:
        plain_seq.append(list[index])

        index += 1

    for i in range(value_len):
        if current_bit & signs_value != 0:
            plain_seq.append(list[index] * (-1))
        else:
            plain_seq.append(list[index])

        current_bit >>= 1

        index += 1

    return (plain_seq)


def simple_rle(vector):  # simple stupid rle

    rle_result = []

    src = vector

    cur = src.pop(0)

    count = 1

    for i in vector:
        if i == cur:
            count += 1
        else:
            rle_result.append((count, cur))
            cur = i
            count = 1

    rle_result.append((count, cur))

    print(rle_result)

    return rle_result


def no_border_rle(seq):  # good RLE without border
    print(seq)

    service_byte = 0

    skip_len = 1
    rep_len = 1

    rle_result = []
    byte_rle_res = []

    i = 0

    while i < (len(seq) - 1):

        if seq[i] != seq[i + 1]:

            skip_len += 1

            if rep_len != 1:
                service_byte = (1 << 7) + rep_len

                byte_rle_res.append(service_byte)
                byte_rle_res.append(seq[i - 1])

                rle_result.append((1, rep_len, seq[i - 1]))

                skip_len -= 1

                rep_len = 1

                service_byte = 0
        else:
            rep_len += 1

            if skip_len != 1:
                service_byte += skip_len - 1

                byte_rle_res.append(service_byte)
                # byte_rle_res.append(seq[(i-skip_len+1):i])
                for j in range(skip_len - 1):
                    byte_rle_res.append(seq[i - skip_len + 1 + j])

                rle_result.append((0, skip_len - 1, seq[(i - skip_len + 1):i]))

                skip_len = 1

                service_byte = 0

        # print("i: ", i, ". Rep_len: ", rep_len, ". Skip_len: ", skip_len, ". RLE: ", rle_result)


        i += 1

    if rep_len != 1:
        rle_result.append((1, rep_len, seq[i - 1]))

        service_byte = (1 << 7) + rep_len
        byte_rle_res.append(service_byte)
        byte_rle_res.append(seq[i - 1])

    else:
        rle_result.append((0, skip_len, seq[(i - skip_len + 1):]))

        service_byte += skip_len

        byte_rle_res.append(service_byte)
        for j in range(skip_len):
            byte_rle_res.append(seq[i - skip_len + 1 + j])

    # print("i: ", i, ". Rep_len: ", rep_len, ". Skip_len: ", skip_len)

    print(rle_result)
    print(byte_rle_res)
    print("Rle seq len: ", len(byte_rle_res))

    return rle_result


def rle(seq, border, byte_per_sampl):  # good RLE with border
    service_byte = 0

    skip_len = 1
    rep_len = 1

    rle_result = []
    byte_rle_res = []

    i = 0

    while i < (len(seq) - 1):

        if seq[i] != seq[i + 1]:

            skip_len += 1

            if rep_len >= border:
                service_byte = (1 << (byte_per_sampl * 8 - 1)) + rep_len

                byte_rle_res.append(service_byte)
                byte_rle_res.append(seq[i - 1])

                rle_result.append((1, rep_len, seq[i - 1]))

                skip_len -= 1

                rep_len = 1

                service_byte = 0
            else:
                skip_len += rep_len - 1

                rep_len = 1
        else:
            rep_len += 1

            if rep_len >= border:
                if skip_len != 1:
                    service_byte += skip_len - 1

                    byte_rle_res.append(service_byte)

                    for j in range(skip_len - 1):
                        byte_rle_res.append(seq[i - skip_len + 1 + j - (border - 2)])

                    rle_result.append((0, skip_len - 1, seq[(i - skip_len + 1 - (border - 2)):i - (border - 2)]))

                    skip_len = 1

                    service_byte = 0

        i += 1

    if rep_len >= border:

        service_byte = (1 << (byte_per_sampl * 8 - 1)) + rep_len

        rle_result.append((1, rep_len, seq[i - 1]))

        byte_rle_res.append(service_byte)
        byte_rle_res.append(seq[i - 1])

    else:
        skip_len += rep_len - 1

        rle_result.append((0, skip_len, seq[(i - skip_len + 1):]))

        service_byte += skip_len

        byte_rle_res.append(service_byte)
        for j in range(skip_len):
            byte_rle_res.append(seq[i - skip_len + j + 1])

    return byte_rle_res


def rle_decoder(rle_seq, byte_per_sample):
    seq = []

    index = 0

    oldest_bit = 1 << byte_per_sample * 8 - 1

    # mask = (1 << (byte_per_sample * 8 - 1)) - 1
    mask = oldest_bit - 1

    while index < (len(rle_seq) - 1):

        if rle_seq[index] > 65535:
            print("Sorry, but WTF?!")

        # if (rle_seq[index] & 32768) == 0:
        if (rle_seq[index] & oldest_bit) == 0:

            for i in range(rle_seq[index]):
                seq.append(rle_seq[index + i + 1])

            index = index + rle_seq[index] + 1

        else:

            for j in range(rle_seq[index] & mask):
                seq.append(rle_seq[index + 1])

            index += 2

    # print("Seq: ",seq)

    return seq


def arithmetic_coder(seq, byte_count):
    import operator

    seq = [1, 2, 3, 4, 1, 2, 5, 2, 3, 6]

    freq_table = {}

    # max_val = (1 << (8 * byte_count)) - 1

    max_val = 10

    i = max_val * (-1)

    while i < (max_val + 1):
        freq_table.update({i: 0})

        i += 1

    for i in seq:
        freq_table[i] += 1

    seq_len = len(seq)

    j = max_val * (-1)

    max_v = 0

    while j < (max_val + 1):

        if freq_table[j] != 0:
            freq_table[j] = freq_table[j] / seq_len
            # freq_table[j] = freq_table[j] / seq_len

            if max_v < freq_table[j]:
                max_v = freq_table[j]
        else:
            freq_table.pop(j)

        j += 1

    print("Before sorter: ", freq_table)

    sorted_freq_table = sorted(freq_table.items(), key=operator.itemgetter(1), reverse=True)

    high = 0

    for i in range(len(sorted_freq_table)):
        low = high
        high = high + sorted_freq_table[i][1]
        # high += sorted_freq_table[i][1]

        interval = (low, high)

        sorted_freq_table[i] = sorted_freq_table[i].__add__(interval)

    print("Sorted freq_table (list): ", sorted_freq_table)

    interval_dict = {}

    for i in range(len(sorted_freq_table)):
        interval_dict.update({sorted_freq_table[i][0]: sorted_freq_table[i][2:]})

    # interval_dict = {2 : (0, 0.3), 1 : (0.3, 0.5), 3 : (0.5, 0.7), 5 : (0.7, 0.8), 6 : (0.8, 0.9), 4 : (0.9, 1.0)}

    print("Sorted table (dict): ", interval_dict)

    low = 0
    high = sorted_freq_table[len(sorted_freq_table) - 1][3]

    interval = (low, high)

    print("Main interval: ", interval)

    index = seq[1]

    for i in range(len(seq) - 1):
        tmp_diff = high - low

        tmp_mul_l = tmp_diff * interval_dict[seq[i]][0]

        tmp_mul_r = tmp_diff * interval_dict[seq[i]][1]

        new_low = low + tmp_mul_l

        new_high = low + tmp_mul_r

        print("i = ", i, ": ", (new_low, new_high))

        low = new_low
        high = new_high

    coder_result = low

    print(low)

    # decoder part

    decoder_result = []

    code = coder_result
    # print(code)

    for i in range(15):
        for j in sorted_freq_table:
            if (code >= j[2]) and (code < j[3]):
                decoder_result.append(j[0])
                low = j[2]
                high = j[3]
                break

        code = (code - low) / (high - low)

        print("Code ", i, ": ", code)

    print(decoder_result)


def write_bit(bit, result):
    print("Big index: ", bits_count[0])

    if bits_count[0] == 8:
        # write byte_current in file
        print("Must be written: -------------------------- ", byte_current[0])

        bits_count[0] = 0

        byte_current[0] = 0

    if bit == 0:
        byte_current[0] <<= 1
        byte_current[0] |= 0

        bits_count[0] += 1

        bit_invert = 1

        result.append(bit)
    else:
        byte_current[0] <<= 1
        byte_current[0] |= 1

        bits_count[0] += 1

        bit_invert = 0

        result.append(bit)

    while throw_bits[0] > 0:
        if bits_count[0] == 8:
            # write byte_current in file

            print("Must be written: -------------------------- ", byte_current[0])

            bits_count[0] = 0

            byte_current[0] = 0

        byte_current[0] <<= 1

        byte_current[0] |= bit_invert

        bits_count[0] += 1

        result.append(bit_invert)

        throw_bits[0] -= 1

    print(result)

    return byte_current[0]

    # сделать запись последнего байта


def integer_arithmetic_coder(seq, byte_count):
    import operator
    result = []

    seq = [1, 2, 3, 4, 5, 5, 4, 3, 2, 1, 2, 3, 4, 1, 2, 3, 4, 3, 2]

    byte_current = [0]

    # forming the symbols count table

    table = {}

    max_val = (1 << 8 * byte_count) - 1

    min_val = max_val * (-1)

    i = min_val

    while i < (max_val + 1):
        table.update({i: 0})

        i += 1

    for i in seq:
        table[i] += 1

    index = min_val

    while index < (max_val + 1):
        if table[index] == 0:
            table.pop(index)

        index += 1

    print(table)

    sorted_table = sorted(table.items(), key=operator.itemgetter(1), reverse=True)

    high_old = 0

    for i in range(len(sorted_table)):
        sorted_table[i] = sorted_table[i].__add__((sorted_table[i][1] + high_old,))

        high_old = sorted_table[i][1] + high_old

    sorted_table.insert(0, (0, 0, 0))

    # print("Sorted table: ", sorted_table)

    full_table = {}

    for i in range(len(sorted_table)):  # он начал добавлять нулевую строку, что меняет его. Убрать?
        full_table.update({sorted_table[i][0]: (i, sorted_table[i][1], sorted_table[i][2])})

    # full_table = {0 : (0,0,0), 2: (1,3,3), 1: (2, 2, 5), 3: (3,2,7), 5: (4,1,8), 6: (5, 1, 9), 4: (6, 1, 10)}

    print("Full table: ", full_table)

    # coder part

    low = 0

    high = max_val

    divider = sorted_table[len(sorted_table) - 1][2]

    first_qtr = (high + 1) >> 2
    half = first_qtr << 1
    third_qtr = half + first_qtr

    # throw_bits = 0

    index = 0
    i = 0

    while index < len(seq):
        j = full_table[seq[index]][0]
        i += 1

        print(j)

        low_current = low + sorted_table[j - 1][2] * (high - low + 1) // divider
        high_current = low + sorted_table[j][2] * (high - low + 1) // divider - 1

        print("Step ", i, ". Low: ", low_current)
        print("Step ", i, ". High: ", high_current)

        # high = high_current
        # low = low_current
        val_cur = 0

        while 1 != 2:
            if high_current < half:
                val_cur = write_bit(0, result)  # запись бита 0 в файл

                print("Current byte: ", val_cur)


            else:
                if low_current >= half:
                    val_cur = write_bit(1, result)

                    low_current -= half
                    high_current -= half

                    print("Current byte: ", val_cur)


                else:
                    if (low_current >= first_qtr) and (high_current < third_qtr):
                        throw_bits[0] += 1

                        low_current -= first_qtr
                        high_current -= first_qtr

                        print("Current byte: ", val_cur)


                    else:

                        print("Current byte: ", val_cur)

                        break

            low_current += low_current
            high_current += high_current + 1

        # print(j)
        print("Step ", i, ". Normalized low: ", low_current)
        print("Step ", i, ". Normalized high: ", high_current)
        print()

        high = high_current
        low = low_current

        index += 1

    print(result)

    if bits_count[0] > 0:
        print(bits_count)
        print("Must be written: -------------------------- ", val_cur)

    # decoder part

    print("DECODER PART:")

    print(sorted_table)

    value = 49532

    low = 0

    high = max_val

    divider = sorted_table[len(sorted_table) - 1][2]

    first_qtr = (high + 1) >> 2
    half = first_qtr << 1
    third_qtr = half + first_qtr

    coder_result = []
    coder_result.extend(result)

    freq = ((value - low + 1) * divider - 1) // (high - low + 1)
    print("Freq: ", freq)

    j = 1
    k = 0

    while sorted_table[j][2] <= freq:
        j += 1

    low_current = low + sorted_table[j - 1][2] * (high - low + 1) // divider
    high_current = low + sorted_table[j][2] * (high - low + 1) // divider - 1

    while 1 != 2:
        if high_current < half:
            print("Dratuti")
        else:
            if low_current >= half:

                low_current -= half
                high_current -= half

                value -= half

            else:
                if (low_current >= first_qtr) and (high_current < third_qtr):

                    low_current -= first_qtr
                    high_current -= first_qtr

                    value -= first_qtr

                else:
                    break

        low_current += low_current
        high_current += high_current + 1

        value += value + result[16 + k]
        # value <<= 1
        # value |= result[16+k]

        print("Almost plain: ", value)

        k += 1


def rice_coder(seq):
    print("RICE CODER PART")

    result = []
    #rice_power = []

    rice_power.append(int(log(seq[0], 2)))

    code = ""

    cur_code = ""

    mod_mask = 1

    for i in range(rice_power[0] - 1):
        mod_mask <<= 1

        mod_mask |= 1

    for i in range(len(seq)):
        remainder = seq[i] & mod_mask
        quotient = seq[i] >> rice_power[0]

        str_format = "{0:0" + str(rice_power[0]) + "b}"

        cur_code += "0" * quotient + "1" + str_format.format(remainder)

        code += cur_code

        cur_code = ""

    pad_len = 8 - len(code) % 8

    code += "0" * pad_len

    pad_info = "{0:08b}".format(pad_len)
    code = pad_info + code

    b = []

    for i in range(0, len(code), 8):
        byte = code[i: i + 8]
        b.append(int(byte, 2))

    #o_file = open("rice_code.bin", 'wb')
    #o_file.write(b''.join(map(lambda x: pack("<B", x), b)))
    #o_file.close()

    return b


def rice_decoder(code):
    print("DECODER PART")

    # print("Code from decoder: ", code)

    result = []

    pad_info = code[:8]
    pad_len = int(pad_info, 2)

    code = code[8:]
    code = code[:-1 * pad_len]

    code_len = len(code)

    # print(code, code_len)

    index = 0

    # print(rice_power)

    while index < code_len:
        # print("Step")
        # print("Was: ", index)

        zero_count = 0

        while code[index] == '0':
            zero_count += 1

            index += 1

        # print("Zero count: ", zero_count)

        index += 1  # skip service bit

        rem_cur = int(code[index:index + rice_power[0]], 2)  # something is wrong here

        # print("Rem_cur: ", rem_cur)

        result.append(zero_count * (1 << (rice_power[0])) + rem_cur)

        index += rice_power[0]

        # print("Become: ", index)

        # print(result)

        # print()

    return result


def scatterplot(x_data, l_data, r_data, x_label="", y_label="", title="", color="r", yscale_log=False):
    _, ax = plot.subplots()

    ax.scatter(x_data, l_data, s=5, color=color, alpha=0.75)

    ax.scatter(x_data, r_data, s=5, color="b", alpha=0.75)

    if yscale_log == True:
        ax.set_yscale('log')

    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    plot.show()


def four_args_scatterplot(x, y1, y2, y3, y4, x_label="Files", y_label="Coefficients", title="", color="black", yscale_log=False):
    _, ax = plot.subplots()

    ax.scatter(x, y1, s=1, color="black", alpha=0.75)
    ax.scatter(x, y2, s=1, color="red", alpha=0.75)
    ax.scatter(x, y3, s=1, color="green", alpha=0.75)
    ax.scatter(x, y4, s=1, color="blue", alpha=0.75)

    if yscale_log == True:
        ax.set_yscale('log')

    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    plot.show()


def lineplot(x, y_left, y_right, x_label="", y_label="", title=""):
    # Create the plot object
    _, ax = plot.subplots()

    # Plot the best fit line, set the linewidth (lw), color and
    # transparency (alpha) of the line
    ax.plot(x, y_left, lw = 1, color = 'red', alpha = 1)
    ax.plot(x, y_right, lw = 1, color = 'blue', alpha = 1)

    ax.legend(('Left', 'Right'))

    # Label the axes and provide a title
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    plot.show()


def four_args_lineplot(x, y1, y2, y3, y4, x_label="Files", y_label="Coefficients", title="Predictor test"):
    # Create the plot object
    _, ax = plot.subplots()

    # Plot the best fit line, set the linewidth (lw), color and
    # transparency (alpha) of the line
    ax.plot(x, y1, lw = 1, color = 'black', alpha = 1)
    ax.plot(x, y2, lw = 1, color = 'blue', alpha = 1)
    ax.plot(x, y3, lw = 1, color = 'red', alpha = 1)
    ax.plot(x, y4, lw = 1, color = 'green', alpha = 1)

    ax.legend(('First', 'Second', 'Third', 'Fourth'))

    # Label the axes and provide a title
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    plot.show()


def groupedbarplot(x_data, y_data_list, colors, graph_legend, x_label="", y_label="", title=""):
    _, ax = plot.subplots()
    # Total width for all bars at one x location
    total_width = 2
    # Width of each individual bar
    ind_width = total_width / len(y_data_list)
    # This centers each cluster of bars about the x tick mark
    alteration = np.arange(-(total_width/2), total_width/2, ind_width)

    # Draw bars, one category at a time
    for i in range(0, len(y_data_list)):
        # Move the bar to the right on the x-axis so it doesn't
        # overlap with previously drawn ones
        #ax.bar(x_data + alteration[i], y_data_list[i], color = colors[i], label ='', width = ind_width)
        ax.tick_params(labelsize = 15)

        ax.bar(x_data, y_data_list[i], color = colors[i], label ='lala', width = ind_width)

    ax.legend(graph_legend, fontsize = 20)

    ax.set_ylabel(y_label, fontsize = 22)
    ax.set_xlabel(x_label, fontsize = 22)
    ax.set_title(title, fontsize = 22)
#    ax.legend(loc = 'upper right')

    plot.show()


def diff_test(l, r):
    # print("Find max diff value in L and R.")
    # print("-"*30)

    l_max_diff = 0
    r_max_diff = 0

    l_index = 0
    r_index = 0

    for i in range(len(l) - 1):
        if abs(l[i + 1] - l[i]) > l_max_diff:
            l_max_diff = abs(l[i + 1] - l[i])
            l_index = i

        if abs(r[i + 1] - r[i]) > r_max_diff:
            r_max_diff = abs(r[i + 1] - r[i])
            r_index = i

    # print("L max diff:", l_max_diff)
    # print("R max diff:", r_max_diff)

    return (l_max_diff, l[l_index - 2: l_index + 2]), (r_max_diff, r[r_index - 2: r_index + 2])


#    calculate arithmetic mean
def calculate_arithmetic_mean(array):
    # print("Predictor window: ", array)

    arithmetic_mean = 0

    for i in range(len(array)):
        arithmetic_mean += array[i]

    arithmetic_mean //= len(array)

    return arithmetic_mean


def calculate_error(original, predicted):
    err_block = []

    for i in range(len(original)):
        err_block.append(abs(original[i] - predicted[i]))

    return err_block


def signed_calculate_error(original, predicted):
    err_block = []

    for i in range(len(original)):
        err_block.append(original[i] - predicted[i])

    return err_block


#     Arithmetic mean predictor
#   * original_sb - original sample block
#   * pw_size - predictor window size
#   * flag - 0 or 1. If flag = 1, then all predicted elements shift to the left by one.
def amp(original_sb, pw_size):
    #print("Arithmetic mean predictor")

    predicted_sb = original_sb[:pw_size]  # block of the predicted samples
    #predicted_sb = []

    sb_size = len(original_sb)

    for i in range(sb_size - pw_size ):
        predicted_sb.append(calculate_arithmetic_mean(original_sb[i: pw_size + i]))

        #print(original_sb[i: pw_size + i])

    #print(predicted_sb)

    #if flag:
        #predicted_sb.pop(pw_size)

    #print(predicted_sb)

    return predicted_sb


def rms_predictor(original_sb, pw_size):
    #print("Root Mean Square predictor")

    predicted_sb = original_sb[:pw_size]

    #print(predicted_sb)

    for i in range( len(original_sb) - pw_size ):
        predicted_sb.append( int( sqrt( sum( list( map( lambda x: pow(x,2), original_sb[i : i + pw_size] ) ) ) / pw_size ) ) )

    for i in range(len(original_sb) - pw_size ):
        if original_sb[i + pw_size] < 0:
            #print("Here with ", i)
            predicted_sb[i + pw_size] *= -1

    #print(predicted_sb)
    return predicted_sb


def rmc_predictor(original_sb, pw_size):
    #print("Root Mean Cube predictor")

    predicted_sb = original_sb[:pw_size]

    for i in range( len(original_sb) - pw_size ):
        #print(i)

        predicted_sb.append( int( pow( abs(sum( list( map( lambda x: pow(x, 3), original_sb[i : i+pw_size] ) ) ) / pw_size), 1/3)))

    for i in range(len(original_sb) - pw_size ):
        if original_sb[i + pw_size] < 0:
            #print("Here with ", i)
            predicted_sb[i + pw_size] *= -1

    #print(predicted_sb)
    return  predicted_sb


def shift_amp(original_sb, pw_size):    #bad thing
    predicted_sb = original_sb[:pw_size]  # block of the predicted samples
    #predicted_sb = []

    sb_size = len(original_sb)

    for i in range(sb_size - pw_size):
        predicted_sb.append(calculate_arithmetic_mean(original_sb[i+1: pw_size + i+1]))

        #print(original_sb[i+1: pw_size + i+1])

    #print(predicted_sb)

    #if flag:
        #predicted_sb.pop(pw_size)

    #print(predicted_sb)

    return predicted_sb

# ________________________________________________________________________________
if len(sys.argv) == 1:

    print("Hi! What do you want from me?")
    print("Enter \"1\" for fast tests.")
    print("Enter \"2\" for difference block test.")
    print("Enter \"3\" for graph wav view of scheme test readed from file.")
    print("Enter \"4\" for try to do right.")
    print("Enter \"5\" for new test compression.")
    print("Enter \"6\" for predictor type test.")
    print("Enter \"7\" for pw_size test.")
    print("Enter \"8\" for test between schemes.")
    print("Enter \"9\" for test between wavelet and decorrelation.")
    print("Enter nothing for full unsigned compression.")
    usr_chs = input("So, show me your choice: ")

    if usr_chs == '1':
        print("Fast test")

        file_name = "/home/wb/Документы/course_work/mixed/101.wav"

        input_file = open(file_name, 'rb')

        print("File \"" + file_name + "\" is opened!")

        input_header = input_file.read(44)

        interval = range(43, 39, -1)

        size_of_sample_data = 0

        for i in interval:
            size_of_sample_data <<= 8

            size_of_sample_data |= input_header[i]

        n = 16

        bb_size = calc_block_size(n)

        for i in range(100):

            input_data = input_file.read(bb_size)

            l,r = signed_sample_forming(input_data, 2, 2)

            l_diff, l_aver = stereo_decorrelation(l,r)

            l_plain, r_plain = inverse_decorrelation(l_diff, l_aver)

            if l == l_plain and r == r_plain:
                print(i)
            else:
                print("BAD")

    elif usr_chs == '2':
        print("Difference block test.")
        print("-" * 30)
        print("Difference BEFORE wavelet.")

        left_sample_seq = []
        right_sample_seq = []

        file_in = open("/home/wb/Документы/course_work/bible/1.wav", 'rb')

        wav_header = file_in.read(44)
        bits_per_sample = unpack("<H", wav_header[34:36])[0]

        print("Bits per sample:", bits_per_sample)

        byte_seq = file_in.read(2 ** 16)

        sample_sequence_forming_sep(byte_seq)

        size = len(left_sample_seq)

        # print("L:", left_sample_seq)
        # print("R:", right_sample_seq)

        l_max_diff = 0
        r_max_diff = 0

        for i in range(size - 2):

            l_diff = abs(left_sample_seq[i] - left_sample_seq[i + 1])
            r_diff = abs(right_sample_seq[i] - right_sample_seq[i + 1])

            if l_diff > l_max_diff:
                l_max_diff = l_diff

                l_max_i = i

            if r_diff > r_max_diff:
                r_max_diff = r_diff

                r_max_i = i

        # print(l_max_i, r_max_i)

        # print(left_sample_seq[(l_max_i - 5):(l_max_i+6)])

        print("L max difference: {0} between {1} and {2}".format(l_max_diff, left_sample_seq[l_max_i],
                                                                 left_sample_seq[l_max_i + 1]))
        print("R max difference: {0} between {1} and {2}".format(r_max_diff, right_sample_seq[r_max_i],
                                                                 right_sample_seq[r_max_i + 1]))

        print(left_sample_seq[(l_max_i - 5):(l_max_i + 6)])

        print("Difference AFTER wavelet.")

        l_res, l_flag = wavelet_transform(left_sample_seq, size)
        r_res, r_flag = wavelet_transform(right_sample_seq, size)

        l_max_diff = 0
        r_max_diff = 0

        for i in range(size - 2):

            l_diff = abs(l_res[i] - l_res[i + 1])
            r_diff = abs(r_res[i] - r_res[i + 1])

            if l_diff > l_max_diff:
                l_max_diff = l_diff

                l_max_i = i

            if r_diff > r_max_diff:
                r_max_diff = r_diff

                r_max_i = i

        # print(l_max_i, r_max_i)

        print(l_res[(l_max_i - 5):(l_max_i + 6)])

        print("L max difference: {0} between {1} and {2}".format(l_max_diff, l_res[l_max_i], l_res[l_max_i + 1]))
        print("R max difference: {0} between {1} and {2}".format(r_max_diff, r_res[r_max_i], r_res[r_max_i + 1]))

    elif usr_chs == '3':
        file_name = "/home/wb/Документы/course_work/hi_res/1.wav"  # file name from which will be read

        first = [ 2.4456, 2.1813, 2.1739, 1.9614, 2.3042, 2.0893, 2.0677, 2.2691, 2.1201, 2.2114, 2.2628, 2.1864, 2.1642, 2.1021, 2.1559, 2.2413, 2.0739, 2.1296, 2.3668, 1.8168, 2.0713, 2.0938, 2.1986, 2.0840, 2.3164, 2.1120, 2.1270, 2.1193, 1.8810, 2.1590, 2.2730, 1.8870, 1.7012, 1.9990, 1.8504, 1.8147, 1.9184, 1.8261, 1.9842, 1.8579, 1.9306]
        second = [ 2.3857, 2.1514, 2.1435, 1.8972, 2.2753, 2.0634, 2.0362, 2.2395, 2.0856, 2.1858, 2.1973, 2.1722, 2.1373, 2.0735, 2.1112, 2.1881, 1.9939, 2.0855, 2.3289, 1.8672, 2.0425, 2.0617, 2.1806, 2.0494, 2.2770, 2.0270, 2.1011, 2.0865, 1.8134, 2.1330, 2.2405, 1.7615, 1.6099, 1.8321, 1.7687, 1.6768, 1.7554, 1.7180, 1.8040, 1.7225, 1.7893]
        third = [ 2.2304, 2.0317, 2.0226, 1.7762, 2.1446, 1.9503, 1.9079, 2.1139, 1.9705, 2.0607, 2.0593, 2.0536, 2.0170, 1.9567, 1.9881, 2.0572, 1.8672, 1.9627, 2.1947, 1.7370, 1.9343, 1.9473, 2.0680, 1.9398, 2.1486, 1.8935, 1.9926, 1.9706, 1.7067, 2.0185, 2.1185, 1.6407, 1.5175, 1.6828, 1.6494, 1.5731, 1.6304, 1.6221, 1.6703, 1.6058, 1.6666]
        fourth = [ 2.0745, 1.9008, 1.8917, 1.6477, 2.0034, 1.8275, 1.7617, 1.9810, 1.8454, 1.9242, 1.8987, 1.9210, 1.8871, 1.8311, 1.8545, 1.9203, 1.7310, 1.8330, 2.0454, 1.6256, 1.8160, 1.8254, 1.9416, 1.8160, 2.0054, 1.7525, 1.8698, 1.8471, 1.5900, 1.8909, 1.9841, 1.5265, 1.4207, 1.5644, 1.5277, 1.4739, 1.5232, 1.5101, 1.5599, 1.4995, 1.5512]

        #y_list = [first, second, third, fourth]

        time = [ (i+1) for i in range(30)]

        print(time)

        #tmp1 = [ randint(5000000,40000000) for i in range(20)]

        #o_file = open("/home/wb/Документы/course_work/hi_res/test.txt", 'wb')

        #dump(y_list, o_file)

        #o_file.close()

        file_count = 200

        orig_size_list = []

        for i in range(file_count):
            name = "/home/wb/Документы/course_work/mixed/" + str(i+1) + ".wav"

            orig_size = os.path.getsize(name)

            orig_size_list.append(orig_size)

        coeff_sch1 = []
        coeff_sch2 = []
        coeff_sch3 = []
        coeff_sch4 = []

        coeff_amp = []
        coeff_rms = []
        coeff_rmc = []

        i_file = open("/home/wb/Документы/course_work/mixed/test/scheme1.txt", 'rb')

        scheme1_sizes = load(i_file)

        i_file.close()

        i_file = open("/home/wb/Документы/course_work/mixed/test/scheme2.txt", 'rb')

        scheme2_sizes = load(i_file)

        i_file.close()

        i_file = open("/home/wb/Документы/course_work/mixed/test/scheme3.txt", 'rb')

        scheme3_sizes = load(i_file)

        i_file.close()

        i_file = open("/home/wb/Документы/course_work/mixed/test/scheme4.txt", 'rb')

        scheme4_sizes = load(i_file)

        i_file.close()

        i_file = open("/home/wb/Документы/course_work/mixed/test/amp.txt", 'rb')

        amp_sizes = load(i_file)

        i_file.close()

        i_file = open("/home/wb/Документы/course_work/mixed/test/rms.txt", 'rb')

        rms_sizes = load(i_file)

        i_file.close()

        i_file = open("/home/wb/Документы/course_work/mixed/test/rmc.txt", 'rb')

        rmc_sizes = load(i_file)

        i_file.close()

        for i in range(file_count):
            coeff_sch1.append(float('{:.4f}'.format(orig_size_list[i] / scheme1_sizes[i])))
            coeff_sch2.append(float('{:.4f}'.format(orig_size_list[i] / scheme2_sizes[i])))
            coeff_sch3.append(float('{:.4f}'.format(orig_size_list[i] / scheme3_sizes[i])))
            coeff_sch4.append(float('{:.4f}'.format(orig_size_list[i] / scheme4_sizes[i])))

            coeff_amp.append(float('{:.4f}'.format(orig_size_list[i] / amp_sizes[i])))
            coeff_rms.append(float('{:.4f}'.format(orig_size_list[i] / rms_sizes[i])))
            coeff_rmc.append(float('{:.4f}'.format(orig_size_list[i] / rmc_sizes[i])))

        coeff_sch1 = coeff_sch1[:30]
        coeff_sch2 = coeff_sch2[:30]
        coeff_sch3 = coeff_sch3[:30]
        coeff_sch4 = coeff_sch4[:30]

        #print(min(min(coeff_amp), min(coeff_rms)))
        #print(max(max(coeff_amp), max(coeff_rms)))

        coeff_amp = coeff_amp[100:130]
        coeff_rms = coeff_rms[100:130]
        coeff_rmc = coeff_rmc[100:130]

        #groupedbarplot(time, [coeff_sch1, coeff_sch2, coeff_sch3, coeff_sch4], ['black', 'blue', 'red', 'green'], ['Схема 1', 'Схема 4', 'Схема 3', 'Схема 2'], 'Номер теста', 'Коэффициент сжатия')

        groupedbarplot(time, [coeff_amp, coeff_rms, coeff_rmc], ['black', 'blue', '#ff335e'], ('Предиктор 1', 'Предиктор 2', 'Предиктор 3', 'Предиктор 4'), 'Номер теста', 'Коэффициент сжатия')

        #scatterplot(time, tmp, tmp1)

        title = ['Name', 'Arithmetic', 'Square', 'Cube']
        print('{0:^15}{1:^15}{2:^15}{3:^15}'.format(title[0], title[1], title[2], title[3]))

        for i in range(file_count):
            print('{0:^15}{1:^15.4f}{2:^15.4f}{3:^15.4f}'.format(time[i], coeff_amp[i], coeff_rms[i], coeff_rmc[i]))

    elif usr_chs == '4':

        for i in range(1):

            name = str(i + 1) + ".wav"

            #file_name = "/home/wb/Документы/course_work/hi_res/" + name  # file name from which will be read
            #file_name = "/home/wb/Документы/course_work/bible/" + name  # file name from which will be read
            #file_name = "/home/wb/Документы/course_work/bible/5.wav"  # file name from which will be read
            file_name = "/home/wb/Документы/course_work/electric.wav"  # file name from which will be read

            in_file = open(file_name, 'rb')

            print("File \"" + file_name + "\" is opened!")

            #file_name = 'crystallize.wav'

            #in_file = open('/home/wb/Документы/course_work/' + file_name, 'rb')

            wav_header = in_file.read(44)

            interval = range(43, 39, -1)

            size_of_sample_data = 0

            for i in interval:
                size_of_sample_data <<= 8

                size_of_sample_data |= wav_header[i]

            n = 16
            bb_size = calc_block_size(n)  # byte block size

            iter_count = size_of_sample_data // bb_size

            chn_count = 2  # count of channels
            chn_count = 2  # count of channels
            byte_per_sample = 2


            #border = len(l_chan) >> 1
            border = 500

            mean_err_list_l = []
            mean_err_list_r = []

            iter_count = 1

            for k in range(iter_count):
                #print("Step ", k)

                current_bb = in_file.read(bb_size)  # current byte block

                l_chan, r_chan = signed_sample_forming(current_bb, chn_count, byte_per_sample)

                #print("L: ", l_chan)
                #print("R: ", r_chan)

                for i in range(border):

                    pw_size = i+3

                    #print("Predictor window size: ", pw_size)

                    predicted_signal_l = amp(l_chan, pw_size)
                    predicted_signal_r = amp(r_chan, pw_size)

                    #print("Predicted L: ", predicted_signal_l, len(predicted_signal_l))
                    #print("Predicted L shift : ", predicted_signal, len(predicted_signal))
                    #print("Predicted R: ", predicted_signal_r)

                    err_l = calculate_error(l_chan, predicted_signal_l)
                    err_r = calculate_error(r_chan, predicted_signal_r)

                    mean_err_list_l.append(calculate_arithmetic_mean(err_l[pw_size:]))
                    mean_err_list_r.append(calculate_arithmetic_mean(err_r[pw_size:]))

                    #print("Error L: ", err_l)
                    #print("Error R: ", err_r)

                    #print(mean_err_list_l)
                    #print(mean_err_list_r)

            #print("\nMax err list: ", max_err_list)

                x = [i+3 for i in range(border)]

                scatterplot(x, mean_err_list_l, mean_err_list_r)

            in_file.close()

    elif usr_chs == '5':
        print("New test")

        file_in = open("/home/wb/Документы/course_work/crystallize.wav", 'rb')

        wav_header = file_in.read(44)
        bits_per_sample = unpack("<H", wav_header[34:36])[0]

        # print("Bits per sample:", bits_per_sample)

        n = 17

        byte_seq = file_in.read(2 ** n)

        # print("Byte seq:", byte_seq)

        l_chan, r_chan = signed_sample_forming(byte_seq, 2, 2)

        # print("L:", left_sample_seq)
        # print("R:", right_sample_seq)

        l_res, l_flag = signed_wavelet_transform(l_chan, n - 2)
        r_res, r_flag = signed_wavelet_transform(r_chan, n - 2)

        l_plain = signed_inverse_wavelet_transform(l_res, l_flag, n - 2)
        r_plain = signed_inverse_wavelet_transform(r_res, r_flag, n - 2)

        # print("W(L):", l_res)
        # print("W(R):", r_res)

        # print("Plain L:", l_plain)
        # print("Plain R:", r_plain

    elif usr_chs == '6':
        i_file_names = []
        amp_coeff = []
        rms_coeff = []
        rmc_coeff = []

        size_list_amp = []
        size_list_rms = []
        size_list_rmc = []

        pw_size = 2

        file_count = 200
        file_dir = "/home/wb/Документы/course_work/mixed/"

        for i in range(file_count):
            name = str(i+1) + ".wav"

            i_file_names.append(name)

            file_name = file_dir + name

            input_file = open(file_name, 'rb')

            print("File \"" + file_name + "\" is opened!")

            input_header = input_file.read(44)

            interval = range(43, 39, -1)

            size_of_sample_data = 0

            for i in interval:
                size_of_sample_data <<= 8

                size_of_sample_data |= input_header[i]

            n = 16

            predictor_type = 'amp'

            bb_size = calc_block_size(n)

            iter_count = size_of_sample_data // bb_size

            output_file = open("/home/wb/Документы/course_work/Compression_results/signed_test.wav", 'wb')

            output_file.write(input_header)

            # code_file = open("coder_result.bin", 'wb')
            left_code = open("HUFF_L_CHAN.bin", 'wb')
            right_code = open("HUFF_R_CHAN.bin", 'wb')

            signed_sep_compression(bb_size, iter_count, n, predictor_type, pw_size)

            left_code.close()
            right_code.close()

            input_file.close()
            output_file.close()

            l_size = os.path.getsize("/home/wb/PycharmProjects/course_work/HUFF_L_CHAN.bin")
            r_size = os.path.getsize("/home/wb/PycharmProjects/course_work/HUFF_R_CHAN.bin")

            huffman_size = l_size + r_size + 44
            size_list_amp.append(huffman_size)

            comp_rate = (size_of_sample_data + 44) / huffman_size

            print("Compression rate for file \"", file_name, "\": ", comp_rate)

            amp_coeff.append(comp_rate)

        o_file = open("/home/wb/Документы/course_work/mixed/test/amp2.txt", 'wb')

        dump(size_list_amp, o_file)

        o_file.close()

        for i in range(file_count):
            name = str(i + 1) + ".wav"

            file_name = file_dir + name

            input_file = open(file_name, 'rb')

            print("File \"" + file_name + "\" is opened!")

            input_header = input_file.read(44)

            interval = range(43, 39, -1)

            size_of_sample_data = 0

            for i in interval:
                size_of_sample_data <<= 8

                size_of_sample_data |= input_header[i]

            n = 16

            predictor_type = 'rms'

            bb_size = calc_block_size(n)

            iter_count = size_of_sample_data // bb_size

            output_file = open("/home/wb/Документы/course_work/Compression_results/signed_test.wav", 'wb')

            output_file.write(input_header)

            # code_file = open("coder_result.bin", 'wb')
            left_code = open("HUFF_L_CHAN.bin", 'wb')
            right_code = open("HUFF_R_CHAN.bin", 'wb')

            signed_sep_compression(bb_size, iter_count, n, predictor_type, pw_size)

            left_code.close()
            right_code.close()

            input_file.close()
            output_file.close()

            l_size = os.path.getsize("/home/wb/PycharmProjects/course_work/HUFF_L_CHAN.bin")
            r_size = os.path.getsize("/home/wb/PycharmProjects/course_work/HUFF_R_CHAN.bin")

            huffman_size = l_size + r_size + 44
            size_list_rms.append(huffman_size)

            comp_rate = (size_of_sample_data + 44) / huffman_size
            print("Compression rate for file \"", file_name, "\": ", comp_rate)

            rms_coeff.append(comp_rate)

        o_file = open("/home/wb/Документы/course_work/mixed/test/rms2.txt", 'wb')

        dump(size_list_rms, o_file)

        o_file.close()

        for i in range(file_count):
            name = str(i + 1) + ".wav"

            file_name = file_dir + name

            input_file = open(file_name, 'rb')

            print("File \"" + file_name + "\" is opened!")

            input_header = input_file.read(44)

            interval = range(43, 39, -1)

            size_of_sample_data = 0

            for i in interval:
                size_of_sample_data <<= 8

                size_of_sample_data |= input_header[i]

            n = 16

            predictor_type = 'rmc'

            bb_size = calc_block_size(n)

            iter_count = size_of_sample_data // bb_size

            output_file = open("/home/wb/Документы/course_work/Compression_results/signed_test.wav", 'wb')

            output_file.write(input_header)

            # code_file = open("coder_result.bin", 'wb')
            left_code = open("HUFF_L_CHAN.bin", 'wb')
            right_code = open("HUFF_R_CHAN.bin", 'wb')

            signed_sep_compression(bb_size, iter_count, n, predictor_type, pw_size)

            left_code.close()
            right_code.close()

            input_file.close()
            output_file.close()

            l_size = os.path.getsize("/home/wb/PycharmProjects/course_work/HUFF_L_CHAN.bin")
            r_size = os.path.getsize("/home/wb/PycharmProjects/course_work/HUFF_R_CHAN.bin")

            huffman_size = l_size + r_size + 44
            size_list_rmc.append(huffman_size)

            comp_rate = (size_of_sample_data + 44) / huffman_size
            print("Compression rate for file \"", file_name, "\": ", comp_rate)

            rmc_coeff.append(comp_rate)

        o_file = open("/home/wb/Документы/course_work/mixed/test/rmc2.txt", 'wb')

        dump(size_list_rmc, o_file)

        o_file.close()

        title = ['Name', 'Arithmetic', 'Square', 'Cube']
        print('{0:^15}{1:^15}{2:^15}{3:^15}'.format(title[0], title[1], title[2], title[3]))

        for i in range(file_count):
            print('{0:^15}{1:^15.4f}{2:^15.4f}{3:^15.4f}'.format(i_file_names[i], amp_coeff[i], rms_coeff[i], rmc_coeff[i]))

    elif usr_chs == '7':
        in_file = open("/home/wb/Документы/course_work/mixed/105.wav", 'rb')    # 102 неплох

        wav_header = in_file.read(44)

        pw_size = 2
        pw_size_list = []

        pw_size_max = 40

        block_count = 200

        max_err_a_l = [[] for i in range(pw_size_max - 1)]
        max_err_a_r = [[] for i in range(pw_size_max - 1)]

        max_err_s_l = [[] for i in range(pw_size_max - 1)]
        max_err_s_r = [[] for i in range(pw_size_max - 1)]

        max_err_c_l = [[] for i in range(pw_size_max - 1)]
        max_err_c_r = [[] for i in range(pw_size_max - 1)]

        mean_err_a_l = [[] for i in range(pw_size_max - 1)]
        mean_err_a_r = [[] for i in range(pw_size_max - 1)]

        mean_err_s_l = [[] for i in range(pw_size_max - 1)]
        mean_err_s_r = [[] for i in range(pw_size_max - 1)]

        mean_err_c_l = [[] for i in range(pw_size_max - 1)]
        mean_err_c_r = [[] for i in range(pw_size_max - 1)]


        while pw_size <= pw_size_max:
            pw_size_list.append(pw_size)

            pw_size += 1

        index = 0

        while index < block_count:

            print("Step: ", index)

            data = in_file.read(2 ** 16)

            l, r = signed_sample_forming(data, 2, 2)

            pw_size = 2

            while pw_size <= pw_size_max:

                a_l = amp(l, pw_size)
                a_r = amp(r, pw_size)

                s_l = rms_predictor(l, pw_size)
                s_r = rms_predictor(r, pw_size)

                c_l = rmc_predictor(l, pw_size)
                c_r = rmc_predictor(r, pw_size)

                a_err_l = calculate_error(l, a_l)
                a_err_r = calculate_error(r, a_r)

                s_err_l = calculate_error(l, s_l)
                s_err_r = calculate_error(r, s_r)

                c_err_l = calculate_error(l, c_l)
                c_err_r = calculate_error(r, c_r)

                max_err_a_l[pw_size-2].append(max(a_err_l))
                max_err_a_r[pw_size-2].append(max(a_err_r))

                mean_err_a_l[pw_size-2].append(calculate_arithmetic_mean(a_err_l))
                mean_err_a_r[pw_size-2].append(calculate_arithmetic_mean(a_err_r))

                max_err_s_l[pw_size-2].append(max(s_err_l))
                max_err_s_r[pw_size-2].append(max(s_err_r))

                mean_err_s_l[pw_size-2].append(calculate_arithmetic_mean(s_err_l))
                mean_err_s_r[pw_size-2].append(calculate_arithmetic_mean(s_err_r))

                max_err_c_l[pw_size-2].append(max(c_err_l))
                max_err_c_r[pw_size-2].append(max(c_err_r))

                mean_err_c_l[pw_size-2].append(calculate_arithmetic_mean(c_err_l))
                mean_err_c_r[pw_size-2].append(calculate_arithmetic_mean(c_err_r))

                pw_size += 1

            index += 1

        y_a_l = []
        y_a_r = []

        y_s_l = []
        y_s_r = []

        y_c_l = []
        y_c_r = []

        for i in range(len(max_err_a_l)):
            y_a_l.append(calculate_arithmetic_mean(max_err_a_l[i]))
            y_a_r.append(calculate_arithmetic_mean(max_err_a_r[i]))

        for i in range(len(max_err_s_l)):
            y_s_l.append(calculate_arithmetic_mean(max_err_s_l[i]))
            y_s_r.append(calculate_arithmetic_mean(max_err_s_r[i]))

        for i in range(len(max_err_c_l)):
            y_c_l.append(calculate_arithmetic_mean(max_err_c_l[i]))
            y_c_r.append(calculate_arithmetic_mean(max_err_c_r[i]))

        mean_y_a_l = []
        mean_y_a_r = []

        mean_y_s_l = []
        mean_y_s_r = []

        mean_y_c_l = []
        mean_y_c_r = []

        for i in range(len(mean_err_a_l)):
            mean_y_a_l.append(calculate_arithmetic_mean(mean_err_a_l[i]))
            mean_y_a_r.append(calculate_arithmetic_mean(mean_err_a_r[i]))

        for i in range(len(mean_err_s_l)):
            mean_y_s_l.append(calculate_arithmetic_mean(mean_err_s_l[i]))
            mean_y_s_r.append(calculate_arithmetic_mean(mean_err_s_r[i]))

        for i in range(len(mean_err_c_l)):
            mean_y_c_l.append(calculate_arithmetic_mean(mean_err_c_l[i]))
            mean_y_c_r.append(calculate_arithmetic_mean(mean_err_c_r[i]))


        #lineplot(pw_size_list, y_a_l, y_a_r, 'Размер окна предиктора', 'Максимальная ошибка', 'Предиктор 1')
        #lineplot(pw_size_list, y_s_l, y_s_r, 'Размер окна предиктора', 'Mean max error', 'Root mean square predictor')
        #lineplot(pw_size_list, y_c_l, y_c_r, 'Размер окна предиктора', 'Mean max error', 'Root mean cube predictor')

        groupedbarplot(pw_size_list, [y_a_l, y_a_r], ['red', 'blue'], ['Левый канал', 'Правый канал'], 'Размер окна предиктора', 'Максимальная ошибка', 'Предиктор 1')
        groupedbarplot(pw_size_list, [y_s_l, y_s_r], ['red', 'blue'], ['Левый канал', 'Правый канал'], 'Размер окна предиктора', 'Максимальная ошибка', 'Предиктор 2')
        groupedbarplot(pw_size_list, [y_c_l, y_c_r], ['red', 'blue'], ['Левый канал', 'Правый канал'], 'Размер окна предиктора', 'Максимальная ошибка', 'Предиктор 3')


        #lineplot(pw_size_list, mean_y_a_l, mean_y_a_r, 'Predictor window size', 'Mean mean error', 'Arithmetic mean predictor')
        #lineplot(pw_size_list, mean_y_s_l, mean_y_s_r, 'Predictor window size', 'Mean mean error', 'Root mean square predictor')
        #lineplot(pw_size_list, mean_y_c_l, mean_y_c_r, 'Predictor window size', 'Mean mean error', 'Root mean cube predictor')

        groupedbarplot(pw_size_list, [mean_y_a_l, mean_y_a_r], ['red', 'blue'], ['Левый канал', 'Правый канал'], 'Размер окна предиктора', 'Средняя ошибка', 'Предиктор 1')
        groupedbarplot(pw_size_list, [mean_y_s_l, mean_y_s_r], ['red', 'blue'], ['Левый канал', 'Правый канал'], 'Размер окна предиктора', 'Средняя ошибка', 'Предиктор 2')
        groupedbarplot(pw_size_list, [mean_y_c_l, mean_y_c_r], ['red', 'blue'], ['Левый канал', 'Правый канал'], 'Размер окна предиктора', 'Средняя ошибка', 'Предиктор 3')

        l_file = open("/home/wb/Документы/course_work/test/y_a_l.txt", 'wb')
        r_file = open("/home/wb/Документы/course_work/test/y_a_r.txt", 'wb')

        dump(y_a_l, l_file)
        dump(y_a_r, r_file)

        l_file.close()
        r_file.close()

        l_file = open("/home/wb/Документы/course_work/test/y_s_l.txt", 'wb')
        r_file = open("/home/wb/Документы/course_work/test/y_s_r.txt", 'wb')

        dump(y_s_l, l_file)
        dump(y_s_r, r_file)

        l_file.close()
        r_file.close()

        l_file = open("/home/wb/Документы/course_work/test/y_c_l.txt", 'wb')
        r_file = open("/home/wb/Документы/course_work/test/y_c_r.txt", 'wb')

        dump(y_c_l, l_file)
        dump(y_c_r, r_file)

        l_file.close()
        r_file.close()

    elif usr_chs == '8':
        i_file_names = []

        file_count = 1
        file_dir = "/home/wb/Документы/course_work/mixed/"

        amp_coeff = [ 0 for i in range(file_count) ]
        rms_coeff = []
        rmc_coeff = [ 0 for i in range(file_count) ]

        huffman_coeff = []
        rice_coeff = []
        rle_coeff = []

        size_list = []

        for i in range(file_count):
            name = str(i+1) + ".wav"

            i_file_names.append(name)

            file_name = file_dir + name

            input_file = open(file_name, 'rb')

            print("File \"" + file_name + "\" is opened!")

            input_header = input_file.read(44)

            interval = range(43, 39, -1)

            size_of_sample_data = 0

            for i in interval:
                size_of_sample_data <<= 8

                size_of_sample_data |= input_header[i]

            n = 16

            predictor_type = 'rms'

            bb_size = calc_block_size(n)

            iter_count = size_of_sample_data // bb_size

            output_file = open("/home/wb/Документы/course_work/Compression_results/signed_test.wav", 'wb')

            output_file.write(input_header)

            # code_file = open("coder_result.bin", 'wb')
            left_code = open("HUFF_L_CHAN.bin", 'wb')
            right_code = open("HUFF_R_CHAN.bin", 'wb')

            left_rice = open("RICE_L_CHAN.bin", 'wb')
            right_rice = open("RICE_R_CHAN.bin", 'wb')

            signed_sep_compression(bb_size, iter_count, n, predictor_type, 3)

            left_rice.close()
            right_rice.close()

            left_code.close()
            right_code.close()

            input_file.close()
            output_file.close()

            l_size = os.path.getsize("/home/wb/PycharmProjects/course_work/HUFF_L_CHAN.bin")
            r_size = os.path.getsize("/home/wb/PycharmProjects/course_work/HUFF_R_CHAN.bin")

            huffman_size = l_size + r_size + 44

            l_size_rice = os.path.getsize("/home/wb/PycharmProjects/course_work/RICE_L_CHAN.bin")
            r_size_rice = os.path.getsize("/home/wb/PycharmProjects/course_work/RICE_R_CHAN.bin")

            rice_size = l_size_rice + r_size_rice + 44

            size_list.append(rice_size)

            huffman_comp_rate = (size_of_sample_data + 44) / huffman_size

            rice_comp_rate = (size_of_sample_data + 44) / rice_size
            print("Compression rate for file \"", file_name, "\": ", huffman_comp_rate)

            huffman_coeff.append(huffman_comp_rate)

            rice_coeff.append(rice_comp_rate)

        #title = ['Name', 'Arithmetic', 'Square', 'Cube']
        title = ['Name', 'Huffman', 'Rice', 'RLE']
        print('{0:^15}{1:^15}{2:^15}{3:^15}'.format(title[0], title[1], title[2], title[3]))

        for i in range(file_count):
            #print('{0:^15}{1:^15.4f}{2:^15.4f}{3:^15.4f}'.format(i_file_names[i], amp_coeff[i], rms_coeff[i], rmc_coeff[i]))
            print('{0:^15}{1:^15.4f}{2:^15.4f}{3:^15.4f}'.format(i_file_names[i], huffman_coeff[i], rice_coeff[i], rle_coeff[i]))

       # o_file = open("/home/wb/Документы/course_work/mixed/test/test.txt", 'wb')

        #dump(size_list, o_file)

        #o_file.close()


        print("Scheme 1 test is completed!")

    elif usr_chs == '9':
        i_file_names = []
        wavelet_coeff_amp = []
        decorr_coeff_amp = []

        wavelet_coeff_rms = []
        decorr_coeff_rms = []


        size_list_wavelet_amp = []
        size_list_decorr_amp = []

        size_list_wavelet_rms = []
        size_list_decorr_rms = []

        pw_size = 2

        file_count =  10
        file_dir = "/home/wb/Документы/course_work/mixed/"

        for i in range(file_count):
            name = str(i+101) + ".wav"

            i_file_names.append(name)

            file_name = file_dir + name

            input_file = open(file_name, 'rb')

            print("File \"" + file_name + "\" is opened!")

            input_header = input_file.read(44)

            interval = range(43, 39, -1)

            size_of_sample_data = 0

            for i in interval:
                size_of_sample_data <<= 8

                size_of_sample_data |= input_header[i]

            n = 16

            predictor_type = 'amp'

            bb_size = calc_block_size(n)

            iter_count = size_of_sample_data // bb_size

            output_file = open("/home/wb/Документы/course_work/Compression_results/signed_test.wav", 'wb')

            output_file.write(input_header)

            # code_file = open("coder_result.bin", 'wb')
            left_code = open("HUFF_L_CHAN.bin", 'wb')
            right_code = open("HUFF_R_CHAN.bin", 'wb')

            #signed_sep_compression(bb_size, iter_count, n, predictor_type, pw_size)
            compression_scheme_3(bb_size, iter_count, n, "wavelet")

            left_code.close()
            right_code.close()

            input_file.close()
            output_file.close()

            l_size = os.path.getsize("/home/wb/PycharmProjects/course_work/HUFF_L_CHAN.bin")
            r_size = os.path.getsize("/home/wb/PycharmProjects/course_work/HUFF_R_CHAN.bin")

            huffman_size = l_size + r_size + 44
            size_list_wavelet_amp.append(huffman_size)

            comp_rate_wavelet = (size_of_sample_data + 44) / huffman_size

            wavelet_coeff_amp.append(comp_rate_wavelet)

        o_file = open("/home/wb/Документы/course_work/mixed/test/wavelet_amp.txt", 'wb')

        dump(size_list_wavelet_amp, o_file)

        o_file.close()

        print("Amp wavelet done")


        for i in range(file_count):
            name = str(i+101) + ".wav"

            file_name = file_dir + name

            input_file = open(file_name, 'rb')

            print("File \"" + file_name + "\" is opened!")

            input_header = input_file.read(44)

            interval = range(43, 39, -1)

            size_of_sample_data = 0

            for i in interval:
                size_of_sample_data <<= 8

                size_of_sample_data |= input_header[i]

            n = 16

            predictor_type = 'amp'

            bb_size = calc_block_size(n)

            iter_count = size_of_sample_data // bb_size

            output_file = open("/home/wb/Документы/course_work/Compression_results/signed_test.wav", 'wb')

            output_file.write(input_header)

            # code_file = open("coder_result.bin", 'wb')
            difference_code = open("HUFF_DIFF.bin", 'wb')
            average_code = open("HUFF_AVER.bin", 'wb')

            #basic_scheme(bb_size, iter_count, n, predictor_type, pw_size)
            #compression_scheme_1(bb_size, iter_count, n, predictor_type, pw_size)
            compression_scheme_3(bb_size, iter_count, n, "decorrelation")

            difference_code.close()
            average_code.close()

            input_file.close()
            output_file.close()

            diff_size = os.path.getsize("/home/wb/PycharmProjects/course_work/HUFF_DIFF.bin")
            aver_size = os.path.getsize("/home/wb/PycharmProjects/course_work/HUFF_AVER.bin")

            huff_size = diff_size + aver_size + 44
            size_list_decorr_amp.append(huff_size)

            comp_rate_decorr = (size_of_sample_data + 44) / huff_size

            decorr_coeff_amp.append(comp_rate_decorr)

        o_file = open("/home/wb/Документы/course_work/mixed/test/decorrelation_amp.txt", 'wb')

        dump(size_list_decorr_amp, o_file)

        o_file.close()

        print("Amp decorrelation done")


        for i in range(file_count):
            name = str(i+101) + ".wav"

            file_name = file_dir + name

            input_file = open(file_name, 'rb')

            print("File \"" + file_name + "\" is opened!")

            input_header = input_file.read(44)

            interval = range(43, 39, -1)

            size_of_sample_data = 0

            for i in interval:
                size_of_sample_data <<= 8

                size_of_sample_data |= input_header[i]

            n = 16

            predictor_type = 'rms'

            bb_size = calc_block_size(n)

            iter_count = size_of_sample_data // bb_size

            output_file = open("/home/wb/Документы/course_work/Compression_results/signed_test.wav", 'wb')

            output_file.write(input_header)

            # code_file = open("coder_result.bin", 'wb')
            left_code = open("HUFF_L_CHAN.bin", 'wb')
            right_code = open("HUFF_R_CHAN.bin", 'wb')

            #signed_sep_compression(bb_size, iter_count, n, predictor_type, pw_size)
            compression_scheme_3(bb_size, iter_count, n, "wavelet")

            left_code.close()
            right_code.close()

            input_file.close()
            output_file.close()

            l_size = os.path.getsize("/home/wb/PycharmProjects/course_work/HUFF_L_CHAN.bin")
            r_size = os.path.getsize("/home/wb/PycharmProjects/course_work/HUFF_R_CHAN.bin")

            huffman_size = l_size + r_size + 44
            size_list_wavelet_rms.append(huffman_size)

            comp_rate_wavelet = (size_of_sample_data + 44) / huffman_size

            wavelet_coeff_rms.append(comp_rate_wavelet)

        o_file = open("/home/wb/Документы/course_work/mixed/test/wavelet_rms.txt", 'wb')

        dump(size_list_wavelet_amp, o_file)

        o_file.close()

        print("Rms wavelet done")


        for i in range(file_count):
            name = str(i+101) + ".wav"

            file_name = file_dir + name

            input_file = open(file_name, 'rb')

            print("File \"" + file_name + "\" is opened!")

            input_header = input_file.read(44)

            interval = range(43, 39, -1)

            size_of_sample_data = 0

            for i in interval:
                size_of_sample_data <<= 8

                size_of_sample_data |= input_header[i]

            n = 16

            predictor_type = 'rms'

            bb_size = calc_block_size(n)

            iter_count = size_of_sample_data // bb_size

            output_file = open("/home/wb/Документы/course_work/Compression_results/signed_test.wav", 'wb')

            output_file.write(input_header)

            # code_file = open("coder_result.bin", 'wb')
            difference_code = open("HUFF_DIFF.bin", 'wb')
            average_code = open("HUFF_AVER.bin", 'wb')

            #basic_scheme(bb_size, iter_count, n, predictor_type, pw_size)
            #compression_scheme_1(bb_size, iter_count, n, predictor_type, pw_size)
            compression_scheme_3(bb_size, iter_count, n, "decorrelation")

            difference_code.close()
            average_code.close()

            input_file.close()
            output_file.close()

            diff_size = os.path.getsize("/home/wb/PycharmProjects/course_work/HUFF_DIFF.bin")
            aver_size = os.path.getsize("/home/wb/PycharmProjects/course_work/HUFF_AVER.bin")

            huff_size = diff_size + aver_size + 44
            size_list_decorr_rms.append(huff_size)

            comp_rate_decorr = (size_of_sample_data + 44) / huff_size

            decorr_coeff_rms.append(comp_rate_decorr)

        o_file = open("/home/wb/Документы/course_work/mixed/test/decorrelation_rms.txt", 'wb')

        dump(size_list_decorr_amp, o_file)

        o_file.close()

        print("Rms decorrelation done")


        title = ['Name', 'Wav Arithmetic', 'Dec Arithmetic', 'Wav Square', 'Dec Square']
        print('{0:^15}{1:^15}{2:^15}{3:^15}{4:^15}'.format(title[0], title[1], title[2], title[3], title[4]))

        for i in range(file_count):
            print('{0:^15}{1:^15.4f}{2:^15.4f}{3:^15.4f}{4:^15.4f}'.format(i_file_names[i], wavelet_coeff_amp[i], decorr_coeff_amp[i], wavelet_coeff_rms[i], decorr_coeff_rms[i]))

    else:
        for i in range(1):
            name = str(i + 1) + ".wav"

            rice_power = []

            file_name = "/home/wb/Документы/course_work/mixed/" + name  # file name from which will be read
            # file_name = "/home/wb/Документы/course_work/crystallize.wav"

            input_file = open(file_name, 'rb')

            print("File \"" + file_name + "\" is opened!")

            input_header = input_file.read(44)

            interval = range(43, 39, -1)

            size_of_sample_data = 0

            for i in interval:
                size_of_sample_data <<= 8

                size_of_sample_data |= input_header[i]

            # print("Data size:", size_of_sample_data, "byte")
            # ------------------------------------

            n = 16  # 2^n

            # print(n)
            #---------------------------

            block_size = calc_block_size(n)

            iter_count = size_of_sample_data // block_size

            input_data = input_file.read(block_size)
            input_data = input_file.read(block_size)

            input_data = input_data[:64]

            l_chan, r_chan = signed_sample_forming(input_data, 2, 2)

            l_predicted = rms_predictor(l_chan, 2)

            l_err = calculate_error(l_chan, l_predicted)

            l_wavelet, l_flag = signed_wavelet_transform(l_err, 4)

            print(l_chan)
            print(l_predicted)
            print(l_err)
            print(l_wavelet)
            print(max(l_err))
            print(calculate_arithmetic_mean(l_err))

elif len(sys.argv) == 2:

    file_dir = sys.argv[1]

    file_name = file_dir[file_dir.rfind('/') + 1:]

    print(file_name)

    input_file = open(file_dir, 'rb')

    print("File \"" + file_dir + "\" is opened!")

    input_header = input_file.read(44)

    interval = range(43, 39, -1)

    size_of_sample_data = 0

    for i in interval:
        size_of_sample_data <<= 8

        size_of_sample_data |= input_header[i]

    n = 16

    pw_size = 2

    predictor_type = 'rms'

    bb_size = calc_block_size(n)

    iter_count = size_of_sample_data // bb_size

    left_code = open("/home/wb/PycharmProjects/course_work/HUFF_L_CHAN.bin", 'wb')
    right_code = open("/home/wb/PycharmProjects/course_work/HUFF_R_CHAN.bin", 'wb')

    signed_sep_compression(bb_size, iter_count, n, predictor_type, pw_size)

    left_code.close()
    right_code.close()

    input_file.close()

    l_size = os.path.getsize("/home/wb/PycharmProjects/course_work/HUFF_L_CHAN.bin")
    r_size = os.path.getsize("/home/wb/PycharmProjects/course_work/HUFF_R_CHAN.bin")

    huffman_size = l_size + r_size + 44

    comp_rate = (size_of_sample_data + 44) / huffman_size

    title = ['Name', 'Compression rate', 'Percent']
    print('{0:^15}{1:^15}{2:^15}'.format(title[0], title[1], title[2]))

    print('{0:^15}{1:^15.4f}{2:^15.4f}'.format(file_name, comp_rate, 100/comp_rate))

elif len(sys.argv) == 3:
    print("full")

    out_dir = sys.argv[2]

    file_dir = sys.argv[1]

    file_name = file_dir[file_dir.rfind('/') + 1:]

    print(file_name)

    input_file = open(file_dir, 'rb')

    print("File \"" + file_dir + "\" is opened!")

    input_header = input_file.read(44)

    interval = range(43, 39, -1)

    size_of_sample_data = 0

    for i in interval:
        size_of_sample_data <<= 8

        size_of_sample_data |= input_header[i]

    n = 16

    pw_size = 2

    predictor_type = 'rms'

    bb_size = calc_block_size(n)

    iter_count = size_of_sample_data // bb_size

    left_code = open("/home/wb/PycharmProjects/course_work/HUFF_L_CHAN.bin", 'wb')
    right_code = open("/home/wb/PycharmProjects/course_work/HUFF_R_CHAN.bin", 'wb')

    signed_sep_compression(bb_size, iter_count, n, predictor_type, pw_size)

    left_code.close()
    right_code.close()

    input_file.close()

    l_size = os.path.getsize("/home/wb/PycharmProjects/course_work/HUFF_L_CHAN.bin")
    r_size = os.path.getsize("/home/wb/PycharmProjects/course_work/HUFF_R_CHAN.bin")

    huffman_size = l_size + r_size + 44

    comp_rate = (size_of_sample_data + 44) / huffman_size

    title = ['Name', 'Compression rate', 'Percent']
    print('{0:^15}{1:^15}{2:^15}'.format(title[0], title[1], title[2]))

    print('{0:^15}{1:^15.4f}{2:^15.4f}'.format(file_name, comp_rate, 100/comp_rate))

    o = open("/home/wb/PycharmProjects/course_work/HUFF_L_CHAN.bin", 'wb')

    fff = o.read()



else:
    print("ARGS ERR")