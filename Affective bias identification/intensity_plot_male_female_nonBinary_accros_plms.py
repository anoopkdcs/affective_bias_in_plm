# -*- coding: utf-8 -*-
"""
Created on Sun Apr 17 22:43:45 2022

@author: Anoop K
 Intensity Plot for Male/Female/Non-binary accros plms
    
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
def sent_pair_reader(file1_path,file2_path, file3_path):
    
    # Read Sentence Paires 
    # file structure --> {Sentence, gtruth, prediction, Prediction Intensity}
    #file index --> {Sentence:0, gtruth:1, prediction:2, Prediction Intensity:3}
    file1 = open(file1_path)
    file2 = open(file2_path)
    file3 = open(file3_path)
    
    file1_csv_reader = csv.reader(file1)
    file1_header = next(file1_csv_reader)
    
    file2_csv_reader = csv.reader(file2)
    file2_header = next(file2_csv_reader)
    
    file3_csv_reader = csv.reader(file3)
    file3_header = next(file3_csv_reader)
    
    sent_pair_1 = []
    sent_pair_2 = []
    sent_pair_3 = []
    
    for i in file1_csv_reader:
            sent_pair_1.append(i)    
    sent_pair_1 = np.array(sent_pair_1)  
    
    for j in file2_csv_reader:
            sent_pair_2.append(j)    
    sent_pair_2 = np.array(sent_pair_2)
    
    for j in file3_csv_reader:
            sent_pair_3.append(j)    
    sent_pair_3 = np.array(sent_pair_3)
    
    return sent_pair_1, sent_pair_2, sent_pair_3

def affective_bias_finder(data1, data2, data3, pair1_label, pair2_label, pair3_label, model_name):
    
    # Plot Prediction Intensity 
    x_indices = []
    emo_intensity_list1 = []
    emo_intensity_list2 = []
    emo_intensity_list3 = []
    for i in range(len(data1)):
        if (data1[i][2] == data2[i][2]) & (data1[i][2] == data3[i][2]) & (data2[i][2] == data3[i][2]) & (data1[i][1] == data2[i][1]) & (data2[i][1] == data3[i][1]) & (data2[i][1] == data3[i][1]):
            if (data1[i][1] == data1[i][2]) & (data2[i][1] == data2[i][2]) & (data3[i][1] == data3[i][2]):
                x_indices.append(i)
                emo_intensity_list1.append(np.float64(data1[i][3]))
                emo_intensity_list2.append(np.float64(data2[i][3]))
                emo_intensity_list3.append(np.float64(data3[i][3]))

    
    fig = plt.figure(figsize = (12, 5))
    plt.plot(x_indices,emo_intensity_list1, label = str(pair1_label) + " "+ str(model_name), color ='r', marker='o', linestyle = 'None') 
    plt.plot(x_indices,emo_intensity_list2, label = str(pair2_label) + " "+ str(model_name), color ='b', marker='o', linestyle = 'None') 
    plt.plot(x_indices,emo_intensity_list3, label = str(pair3_label) + " "+ str(model_name), color ='g', marker='o', linestyle = 'None') 
    
    plt.axhline(y = mean(emo_intensity_list1), color = 'r', linestyle = '--', label= str(pair1_label) + ' '+'mean')
    plt.axhline(y = mean(emo_intensity_list2), color = 'b', linestyle = '--', label= str(pair2_label) + ' '+'mean')
    plt.axhline(y = mean(emo_intensity_list3), color = 'g', linestyle = '--', label= str(pair2_label) + ' '+'mean')
        
    plt.xlabel('Input Sentences', fontsize = 12)
    plt.ylabel('Predicted Emotion Intensity', fontsize = 12)
    plt.legend( loc='upper center', bbox_to_anchor=(.5, 1.15),
                  fancybox=True, shadow=True, ncol=6,fontsize=12 )
    plot_name = 'plots/'+str(context)+'/'+str(eval_corpus)+'/'+'intensity_plot_over_plm_mfn_'+str(model_name)+'_'+str(pair1_label)+'_'+str(pair2_label)
    plt.savefig(plot_name+'.png')
    plt.show()
    
    
    
    return
# --------- Main function ------
#Change Variables 
eval_corpus = 'bits'
context = 'gender'
pair1_label = 'male'
pair2_label = 'female'
pair3_label = 'NB'
bert_pair1_path = 'sentence_paires_predictions/gender/bits/bits_male_120_bits_bert_predictions.csv'
bert_pair2_path = 'sentence_paires_predictions/gender/bits/bits_female_120_bits_bert_predictions.csv'
bert_pair3_path = 'sentence_paires_predictions/gender/bits/non_binary_120_bits_bert_predictions.csv'

gpt2_pair1_path = 'sentence_paires_predictions/gender/bits/bits_male_120_bits_gpt2_predictions.csv'
gpt2_pair2_path = 'sentence_paires_predictions/gender/bits/bits_female_120_bits_gpt2_predictions.csv'
gpt2_pair3_path = 'sentence_paires_predictions/gender/bits/non_binary_120_bits_gpt2_predictions.csv'

xlnet_pair1_path = 'sentence_paires_predictions/gender/bits/bits_male_120_bits_XLNet_predictions.csv'
xlnet_pair2_path = 'sentence_paires_predictions/gender/bits/bits_female_120_bits_XLNet_predictions.csv'
xlnet_pair3_path = 'sentence_paires_predictions/gender/bits/non_binary_120_bits_XLNet_predictions.csv'

t5_pair1_path = 'sentence_paires_predictions/gender/bits/bits_male_120_bits_T5_predictions.csv'
t5_pair2_path = 'sentence_paires_predictions/gender/bits/bits_female_120_bits_T5_predictions.csv'
t5_pair3_path = 'sentence_paires_predictions/gender/bits/non_binary_120_bits_T5_predictions.csv'


# Affective Bias finder
bert_pair_1, bert_pair_2, bert_pair_3 = sent_pair_reader(bert_pair1_path, bert_pair2_path, bert_pair3_path)
gpt2_pair_1, gpt2_pair_2, gpt2_pair_3 = sent_pair_reader(gpt2_pair1_path, gpt2_pair2_path, gpt2_pair3_path)
xlnet_pair_1, xlnet_pair_2, xlnet_pair_3 = sent_pair_reader(xlnet_pair1_path, xlnet_pair2_path, xlnet_pair3_path)
t5_pair_1, t5_pair_2, t5_pair_3 = sent_pair_reader(t5_pair1_path, t5_pair2_path, t5_pair3_path)

affective_bias_finder(bert_pair_1, bert_pair_2, bert_pair_3, pair1_label, pair2_label, pair3_label, 'BERT')
affective_bias_finder(gpt2_pair_1, gpt2_pair_2, gpt2_pair_3, pair1_label, pair2_label, pair3_label, 'GPT-2')
affective_bias_finder(xlnet_pair_1, xlnet_pair_2, xlnet_pair_3, pair1_label, pair2_label, pair3_label, 'XLNet')
affective_bias_finder(t5_pair_1, t5_pair_2, t5_pair_3, pair1_label, pair2_label, pair3_label, 'T5')