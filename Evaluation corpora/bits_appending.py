# -*- coding: utf-8 -*-
"""
Created on Mon Apr  4 19:56:37 2022

@author: Anoop K
    
BITS append
"""

import numpy as np 

gender_data_bits = np.load('data/bits/bits_npy_all_data_and_label/gender_data_bits.npy')
gender_label_bits = np.load('data/bits/bits_npy_all_data_and_label/gender_label_bits.npy')

race_data_bits = np.load('data/bits/bits_npy_all_data_and_label/race_data_bits.npy')
race_label_bits = np.load('data/bits/bits_npy_all_data_and_label/race_label_bits.npy')


bits_data = np.concatenate((gender_data_bits,race_data_bits),axis=0)
bits_label = np.concatenate((gender_label_bits,race_label_bits),axis=0)



np.save('data/bits/bits_npy_all_data_and_label/bits_sentences.npy',bits_data)
np.save('data/bits/bits_npy_all_data_and_label/bits_labels.npy', bits_label) 