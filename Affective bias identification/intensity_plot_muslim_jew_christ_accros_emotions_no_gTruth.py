# -*- coding: utf-8 -*-
"""
Created on Mon Apr 18 11:35:25 2022

@author: Anoop K
Intensity Plot for Male/Female/Non-binary accros emotions and no groud truth 
for CSP Data
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
def affective_bias_finder(emotion, pair1_label, pair2_label, pair3_label):

    if emotion == '0':
        emotion_label_text = 'Anger'
        
    if emotion == '1':
        emotion_label_text = 'Fear'
        
    if emotion == '2':
        emotion_label_text = 'Joy'
        
    if emotion == '3':
        emotion_label_text = 'Sadness'
        
        
    
    # # Plot Prediction Intensity  
    pair1_emotion_indices = list(np.where(sent_pair_1[:,1] == emotion))
    x_indices = []
    pair1_emotion_list = []
    pair2_emotion_list = []
    pair3_emotion_list = []
    for i in range(len(pair1_emotion_indices[0])):
        indx = pair1_emotion_indices[0][i]
        if (sent_pair_1[indx][1] == sent_pair_2[indx][1]) & (sent_pair_1[indx][1] == sent_pair_3[indx][1]) & (sent_pair_2[indx][1] == sent_pair_3[indx][1]):
            if (sent_pair_1[indx][1] == emotion) & (sent_pair_2[indx][1] == emotion) & (sent_pair_3[indx][1] == emotion):            
                x_indices.append(indx)
                pair1_emotion_list.append(np.float64(sent_pair_1[indx][2]))
                pair2_emotion_list.append(np.float64(sent_pair_2[indx][2]))
                pair3_emotion_list.append(np.float64(sent_pair_3[indx][2]))
            
    pair1_mean = mean(pair1_emotion_list)
    pair2_mean = mean(pair2_emotion_list)
    pair3_mean = mean(pair3_emotion_list)
    
    fig = plt.figure(figsize = (9, 4))#10, 5
    # insted of x_indices --> range(len(pair1_emotion_list)) for 1 to N x axis values
    plt.plot(x_indices,pair1_emotion_list, label = str(pair1_label) + " "+ str(emotion_label_text), color ='r', marker='o', linestyle = 'None') 
    plt.axhline(y = pair1_mean, color = 'r', linestyle = '--', label= str(pair1_label) + ' '+'Mean')
    
    plt.plot(x_indices, pair2_emotion_list, label = str(pair2_label) +" " +str(emotion_label_text), color ='b', marker='o', linestyle = 'None')
    plt.axhline(y = pair2_mean, color = 'b', linestyle = '--', label= str(pair2_label) + ' '+'Mean')
    
    plt.plot(x_indices, pair3_emotion_list, label = str(pair3_label) +" " +str(emotion_label_text), color ='g', marker='o', linestyle = 'None')
    plt.axhline(y = pair3_mean, color = 'g', linestyle = '--', label= str(pair3_label) + ' '+'Mean')
    
    plt.xlabel('Input Sentences', fontsize = 12)
    plt.ylabel('Predicted Emotion Intensity', fontsize = 12)
    plt.legend( loc='upper center', bbox_to_anchor=(.5, 1.25),#.5, 1.12
              fancybox=True, shadow=True, ncol=3,fontsize=12 )
    plot_name = 'intensity_plot_over_emo_mjc_noGt_'+str(model_name)+'_'+str(emotion_label_text)+'_'+str(pair1_label)+'_'+str(pair2_label)
    #plt.savefig(plot_path+plot_name+'.png')
    plt.show()
    
    return 



# --------- Main function starts here ------
#Change Variables 
eval_corpus = 'csp'
context = 'religion'
model_name = 'GPT-2'
pair1_label_text = 'Muslim'
pair2_label_text = 'Jew'
pair3_label_text = 'Christian'
# file structure --> {Sentence, gtruth, prediction, Prediction Intensity}
#file index --> {Sentence:0, gtruth:1, prediction:2, Prediction Intensity:3}
file1 = open('sentence_paires_predictions/religion/csp/csp_muslim_104_csp_gpt2_predictions.csv')
file2 = open('sentence_paires_predictions/religion/csp/csp_jew_104_csp_gpt2_predictions.csv')
file3 = open('sentence_paires_predictions/religion/csp/csp_christian_104_csp_gpt2_predictions.csv')




# Read Sentence Paires 
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



# Finding Affective Bias 
plot_path = 'plots/'+str(context)+'/'+str(eval_corpus)+'/'
affective_bias_finder('0', pair1_label_text, pair2_label_text, pair3_label_text)
affective_bias_finder('1', pair1_label_text, pair2_label_text, pair3_label_text)
affective_bias_finder('2', pair1_label_text, pair2_label_text, pair3_label_text)
affective_bias_finder('3', pair1_label_text, pair2_label_text, pair3_label_text)
