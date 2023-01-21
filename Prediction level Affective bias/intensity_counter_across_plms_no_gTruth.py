# -*- coding: utf-8 -*-
"""
Created on Wed Apr 27 15:26:09 2022

@author: Anoop K

Intensity counter
    1. # of samples give high intensity to socio-group 1
    2. # of samples give high intensity to socio-group 2
    3. # of samples give equal intensity to socio-group 1 and socio-group 3
    4. average confidence score over emotion 
    
    across PLMs without groundtruth 
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

#Senence pair reader function
def sent_pair_reader(file1_path,file2_path):
    
    # Read Sentence Paires 
    # file structure --> {Sentence, prediction, Prediction Intensity}
    #file index --> {Sentence:0,prediction:1, Prediction Intensity:2}
    file1 = open(file1_path)
    file2 = open(file2_path)
    
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
    
    return sent_pair_1, sent_pair_2


def affective_bias_finder(data1, data2, pair1_label, pair2_label, model_name):
    
    #  Prediction Intensity 
    print("---" +str(model_name)+"---")    
    pair1_high_count = 0
    pair2_high_count = 0
    pair2_equal_pair2_count = 0
    confidence_scores = []
    decimal_val = 3
    
    x_indices = []
    emo_intensity_list1 = []
    emo_intensity_list2 = []
    for i in range(len(data1)):
        if(data1[i][1] == data2[i][1]):
            x_indices.append(i)
            emo_intensity_list1.append(np.float64(data1[i][2]))
            emo_intensity_list2.append(np.float64(data2[i][2]))
            
            #confidence score 
            confidence = 1 - (np.float64(data1[i][2])/np.float64(data2[i][2]))
            confidence_scores.append(confidence)
    
    confidence_scores = np.array(confidence_scores)
    avg_conf_score = np.average(confidence_scores)
    print("Average Confidense score for "+ str(model_name)+ ": ", avg_conf_score)
    
    
    for j in range(len(emo_intensity_list1)):
        p1 = np.round(emo_intensity_list1[j],decimal_val)
        p2 = np.round(emo_intensity_list2[j],decimal_val)
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

# --------- Main function ------
#Change Variables 
#eval_corpus = 'eec'
#context = 'gender'
pair1_label = 'Muslim'
pair2_label = 'Jew'
bert_pair1_path = 'sentence_paires_predictions/religion/csp/csp_muslim_104_csp_bert_predictions.csv'
bert_pair2_path = 'sentence_paires_predictions/religion/csp/csp_jew_104_csp_bert_predictions.csv'

gpt2_pair1_path = 'sentence_paires_predictions/religion/csp/csp_muslim_104_csp_gpt2_predictions.csv'
gpt2_pair2_path = 'sentence_paires_predictions/religion/csp/csp_jew_104_csp_gpt2_predictions.csv'

xlnet_pair1_path = 'sentence_paires_predictions/religion/csp/csp_muslim_104_csp_XLNet_predictions.csv'
xlnet_pair2_path = 'sentence_paires_predictions/religion/csp/csp_jew_104_csp_XLNet_predictions.csv'

t5_pair1_path = 'sentence_paires_predictions/religion/csp/csp_muslim_104_csp_T5_predictions.csv'
t5_pair2_path = 'sentence_paires_predictions/religion/csp/csp_jew_104_csp_T5_predictions.csv'

# Affective Bias finder
bert_pair_1, bert_pair_2 = sent_pair_reader(bert_pair1_path, bert_pair2_path)
gpt2_pair_1, gpt2_pair_2 = sent_pair_reader(gpt2_pair1_path, gpt2_pair2_path)
xlnet_pair_1, xlnet_pair_2 = sent_pair_reader(xlnet_pair1_path, xlnet_pair2_path)
t5_pair_1, t5_pair_2 = sent_pair_reader(t5_pair1_path, t5_pair2_path)

affective_bias_finder(bert_pair_1, bert_pair_2, pair1_label, pair2_label, 'BERT')
affective_bias_finder(gpt2_pair_1, gpt2_pair_2, pair1_label, pair2_label, 'GPT-2')
affective_bias_finder(xlnet_pair_1, xlnet_pair_2, pair1_label, pair2_label, 'XLNet')
affective_bias_finder(t5_pair_1, t5_pair_2, pair1_label, pair2_label, 'T5')