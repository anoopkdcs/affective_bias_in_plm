#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 18 19:17:32 2022

@author: user
"""
import csv
import numpy as np 

file_path = open('data/NRC-Emotion-Intensity-Lexicon-v1.csv')
file_csvreader = csv.reader(file_path,delimiter='\t')
file_header = next(file_csvreader)
file_rows = []

for i in file_csvreader:
        file_rows.append(i)
        
emotion_lexicon = np.array(file_rows)   



def emo_word_list(emotion,delta,emolex):
    emotion_indices = np.where((emolex[:,1] == emotion) & (np.float64(emolex[:,2]) >= delta))
    emotion_data = emolex[emotion_indices[0]]
    emotion_words = emotion_data[:,0].tolist()
    return emotion_words


anger = emo_word_list('anger', 0.1, emotion_lexicon)
fear = emo_word_list('fear', 0.1, emotion_lexicon)
joy = emo_word_list('joy', 0.1, emotion_lexicon)
sadness = emo_word_list('sadness', 0.1, emotion_lexicon)

#total data
total = len(anger) + len(fear) + len(joy) + len(sadness)
