# -*- coding: utf-8 -*-
"""
Created on Wed Apr 27 11:50:13 2022

 @author: Anoop K

Intensity counter
    1. # of samples give high intensity to socio-group 1
    2. # of samples give high intensity to socio-group 2
    3. # of samples give equal intensity to socio-group 1 and socio-group 3
    4. average confidence score over emotion 
    
    across Emotions with groundtruth 
"""

import csv 
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt 
from statistics import mean
from scipy.stats import ttest_ind
from scipy.stats import ks_2samp
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import pyplot


# Affective bias finder function 
def affective_bias_finder(emotion, pair1_label, pair2_label):
    
    pair1_high_count = 0
    pair2_high_count = 0
    pair2_equal_pair2_count = 0
    confidence_scores = []
    decimal_val = 3

    if emotion == '0':
        emotion_label_text = 'anger'
        
    if emotion == '1':
        emotion_label_text = 'fear'
        
    if emotion == '2':
        emotion_label_text = 'joy'
        
    if emotion == '3':
        emotion_label_text = 'sadness'
        
        
    print("---- " + str(emotion_label_text) + " ----")
    # # Prediction Intensity  
    pair1_emotion_indices = list(np.where(sent_pair_1[:,1] == emotion))
    x_indices = []
    pair1_emotion_list = []
    pair2_emotion_list = []
    for i in range(len(pair1_emotion_indices[0])):
        indx = pair1_emotion_indices[0][i]
        if (sent_pair_1[indx][2] == sent_pair_2[indx][2]) & (sent_pair_1[indx][1] == emotion) & (sent_pair_2[indx][1] == emotion) & (sent_pair_1[indx][1] == sent_pair_1[indx][2]) & (sent_pair_2[indx][1] == sent_pair_2[indx][2]):
            x_indices.append(indx)
            pair1_emotion_list.append(np.float64(sent_pair_1[indx][3]))
            pair2_emotion_list.append(np.float64(sent_pair_2[indx][3]))
            
            
            #confidence score 
            confidence = 1 - (np.float64(sent_pair_1[indx][3])/np.float64(sent_pair_2[indx][3]))
            confidence_scores.append(confidence)
            
    confidence_scores = np.array(confidence_scores)
    avg_conf_score = np.average(confidence_scores)
    
    print("Average Confidense score for "+ str(emotion_label_text)+" "+ str(pair1_label)+ " Vs. " + str(pair2_label)+  ": ", avg_conf_score)
    
    
            
    for j in range(len(pair1_emotion_list)):

        p1 = np.round(pair1_emotion_list[j],decimal_val)
        p2 = np.round(pair2_emotion_list[j],decimal_val)
        
        if p1 > p2:
            pair1_high_count = pair1_high_count + 1
        
        if p2 > p1:
            pair2_high_count = pair2_high_count + 1
            
        if p1 == p2 :
            pair2_equal_pair2_count = pair2_equal_pair2_count + 1

    print(str(pair1_label) + " high count: ", pair1_high_count)
    print(str(pair2_label) + " high count: ", pair2_high_count)
    print(str(pair1_label) + " and " + str(pair2_label) + " equal count: ", pair2_equal_pair2_count)
    
    total_inten = pair1_high_count + pair2_high_count + pair2_equal_pair2_count

    pair1_high_count_per = pair1_high_count / total_inten
    pair2_high_count_per = pair2_high_count / total_inten
    pair1_equal_pair2_count_per = pair2_equal_pair2_count / total_inten

    print(str(pair1_label) + " high count in percentage: ", pair1_high_count_per*100)
    print(str(pair2_label) + " high count in percentage: ", pair2_high_count_per*100)
    print(str(pair1_label) + " and " + str(pair2_label) + " equal count in percentage: ", pair1_equal_pair2_count_per*100)

    print("\n")     
    return 



# --------- Main function starts here ------
#Change Variables 
#eval_corpus = 'eec'
#context = 'race'
#model_name = 'GPT-2' #'T5', 'XLNet', 'GPT-2', 'BERT'
pair1_label_text = 'euro american'
pair2_label_text = 'afro american'
# file structure --> {Sentence, gtruth, prediction, Prediction Intensity}
#file index --> {Sentence:0, gtruth:1, prediction:2, Prediction Intensity:3}
file1 = open('sentence_paires_predictions/race/bits/bits_euro_american_120_bits_T5_predictions.csv')
file2 = open('sentence_paires_predictions/race/bits/bits_afri_american_120_bits_T5_predictions.csv')
    

# Read Sentence Paires 
file1_csv_reader = csv.reader(file1)
file1_header = next(file1_csv_reader)

file2_csv_reader = csv.reader(file2)
file2_header = next(file2_csv_reader)

sent_pair_1 = []
sent_pair_2 = []

for i in file1_csv_reader:
        sent_pair_1.append(i)    
sent_pair_1 = np.array(sent_pair_1)  

for j in file2_csv_reader:
        sent_pair_2.append(j)    
sent_pair_2 = np.array(sent_pair_2)


affective_bias_finder('0', pair1_label_text, pair2_label_text)
affective_bias_finder('1', pair1_label_text, pair2_label_text)
affective_bias_finder('2', pair1_label_text, pair2_label_text)
affective_bias_finder('3', pair1_label_text, pair2_label_text)
