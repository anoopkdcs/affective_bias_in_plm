# -*- coding: utf-8 -*-
"""
Created on Sat Apr 16 11:15:58 2022

@author: Anoop K

Affective Bias Finder 
    1. for sentences with out ground truth
    2. along each emotions 
"""

import csv 
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt 
from statistics import mean
from scipy.stats import ttest_rel
from scipy.stats import ks_2samp
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import pyplot


# Affective bias finder function 
def affective_bias_finder(emotion, pair1_label, pair2_label):

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
    for i in range(len(pair1_emotion_indices[0])):
        indx = pair1_emotion_indices[0][i]
        if (sent_pair_1[indx][1] == sent_pair_2[indx][1]) & (sent_pair_1[indx][1] == emotion) & (sent_pair_2[indx][1] == emotion):
            x_indices.append(indx)
            pair1_emotion_list.append(np.float64(sent_pair_1[indx][2]))
            pair2_emotion_list.append(np.float64(sent_pair_2[indx][2]))
            
            
    #print(x_indices)
    #print(pair1_emotion_list)
    
    #pair1_emotion_matrix = sent_pair_1[pair1_emotion_indices[0],3]
    #pair1_emotion_list = pair1_emotion_matrix.tolist()
    #pair1_emotion_list = [float(i) for i in pair1_emotion_list] 
    pair1_mean = mean(pair1_emotion_list)
    
    #pair2_emotion_indices = np.where(sent_pair_2[:,1] == emotion)
    #pair2_emotion_matrix = sent_pair_2[pair2_emotion_indices[0],3]
    #pair2_emotion_list = pair2_emotion_matrix.tolist()
    #pair2_emotion_list = [float(i) for i in pair2_emotion_list] 
    pair2_mean = mean(pair2_emotion_list)
    
    fig = plt.figure(figsize = (9, 4))#10, 5
    # insted of x_indices --> range(len(pair1_emotion_list)) for 1 to N x axis values
    plt.plot(x_indices,pair1_emotion_list, label = str(pair1_label) + " "+ str(emotion_label_text), color ='r', marker='o', linestyle = 'None') 
    plt.axhline(y = pair1_mean, color = 'r', linestyle = '--', label= str(pair1_label) +' '+str(emotion_label_text)+' '+'Mean')
    plt.plot(x_indices, pair2_emotion_list, label = str(pair2_label) +" " +str(emotion_label_text), color ='b', marker='o', linestyle = 'None')
    plt.axhline(y = pair2_mean, color = 'b', linestyle = '--', label= str(pair2_label) + ' '+str(emotion_label_text)+' '+'Mean')
    
    plt.xlabel('Input Sentences', fontsize = 12)
    plt.ylabel('Predicted Emotion Intensity', fontsize = 12)
    plt.legend( loc='upper center', bbox_to_anchor=(.5, 1.24),#.5, 1.12
              fancybox=True, shadow=True, ncol=2,fontsize=12 )
    plot_name = 'intensity_plot_over_emo_noGt_'+str(model_name)+'_'+str(emotion_label_text)+'_'+str(pair1_label)+'_'+str(pair2_label)
    #plt.savefig(plot_path+plot_name+'.png')
    plt.show()
    
    
    
    print("----" + str(emotion_label_text) + "----")
    #Mean Intensity Difference 
    print('Mean Intensity '+ str(pair1_label)+': ', pair1_mean)
    print('Mean Intensity '+ str(pair2_label)+': ', pair2_mean)
    print('Mean Intensity Difference: ', np.absolute(pair1_mean-pair2_mean))
    
    
    # Average delta  for each Emotions
    delta_emotion = np.absolute(np.array(pair1_emotion_list) - np.array(pair2_emotion_list))
    #delta_emotion = np.absolute(np.float64(pair1_emotion_matrix) - np.float64(pair2_emotion_matrix))
    avg_delta_emotion = np.average(delta_emotion)
    print("average delta "+str(emotion_label_text)+ ": ",avg_delta_emotion)
    
    
    # Statistical significance over emotions
    value, pvalue = ttest_rel(pair1_emotion_list, pair2_emotion_list) #, equal_var=False
    print("significance test values for "+ str( pair1_label) + " and " + str(pair2_label)+ " with " + str(emotion_label_text)+": ", value, pvalue)
    
    
    # Demographic Parity
    pair1_emotion_prediction_indices = np.array(np.where(sent_pair_1[:,1]== emotion))
    pair1_emotion_prediction_count = pair1_emotion_prediction_indices.shape[1]
    prob_emotion_pair1= pair1_emotion_prediction_count/sent_pair_1.shape[0]

    pair2_emotion_prediction_indices = np.array(np.where(sent_pair_2[:,1]== emotion))
    pair2_emotion_prediction_count = pair2_emotion_prediction_indices.shape[1]
    prob_emotion_pair2= pair2_emotion_prediction_count/sent_pair_2.shape[0]
    print("Demographic Parity over " +str(pair1_label)+ " vs " +str(pair2_label) +": ", prob_emotion_pair1, '/', prob_emotion_pair2)
    
    # Equal Opportunity
        #No ground truth available 

    
    print("\n")
    return 


def confidence_score_finder():
# Violine Plot for confidence score    
    confidence_scores = []
    emotions = []
    for i in range(len(sent_pair_1)):
        if (sent_pair_1[i][1] == sent_pair_2[i][1]):
            emo = sent_pair_1[i][1]
            score_pair_1 = np.float64(sent_pair_1[i][2])
            score_pair_2 = np.float64(sent_pair_2[i][2])
            confidence = 1 - (score_pair_1/score_pair_2)
            confidence_scores.append(confidence)
            emotions.append(emo)
            
    confidence_scores = np.array(confidence_scores)
    emotions = np.array(emotions)
    c_score_emotion = np.concatenate((np.float64(np.reshape(confidence_scores,
                                                            (len(confidence_scores),1))) ,
                                      np.reshape(emotions,(len(emotions),1))), axis = 1)
    
    for j in range(len(c_score_emotion)):
        if c_score_emotion[j][1] == '0':
            c_score_emotion[j][1]  = 'Anger'
        
        if c_score_emotion[j][1] == '1':
            c_score_emotion[j][1]  = 'Fear'
        
        if c_score_emotion[j][1] == '2':
            c_score_emotion[j][1]  = 'Joy'
        
        if c_score_emotion[j][1] == '3':
            c_score_emotion[j][1]  = 'Sadness'
    

    # Violine plot for confidence scores 
    sns.set_style("whitegrid")
    sns.set_context("notebook")
    fig, ax = pyplot.subplots(dpi = 100)#figsize =(6, 3), 
    xx = sns.violinplot(ax = ax, x = c_score_emotion[:,1] , y = np.float64(c_score_emotion[:,0]),
                        hue = c_score_emotion[:,1])
    xx.set(xticklabels=[])
    plt.xlabel(str(pair1_label_text)+' vs. '+str(pair2_label_text), fontsize = 12)
    plt.legend( loc='upper center', bbox_to_anchor=(.49, 1.15), fancybox=True, shadow=True, ncol=4,fontsize=12 )
    plot_name = 'confidence_plot_over_emo_noGt_'+str(model_name)+'_'+str(pair1_label_text)+'_'+str(pair2_label_text)
    #plt.savefig(plot_path+plot_name+'.png')
    plt.show()
    return 

# --------- Main function ------
#Change Variables 
eval_corpus = 'csp'
context = 'race'
model_name = 'BERT' #BERT, GPT-2, XLNet
pair1_label_text = 'African American'
pair2_label_text =   'European American'
# file structure --> {Sentence, prediction, Prediction Intensity}
#file index --> {Sentence:0, prediction:1, Prediction Intensity:2}
file1 = open('sentence_paires_predictions/race/csp/csp_afri_american_566_csp_T5_predictions.csv')
file2 = open('sentence_paires_predictions/race/csp/csp_euro_american_566_csp_T5_predictions.csv') 

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

# Finding Affective Bias 
plot_path = 'plots/'+str(context)+'/'+str(eval_corpus)+'/'
affective_bias_finder('0', pair1_label_text, pair2_label_text)
affective_bias_finder('1', pair1_label_text, pair2_label_text)
affective_bias_finder('2', pair1_label_text, pair2_label_text)
affective_bias_finder('3', pair1_label_text, pair2_label_text)
confidence_score_finder()

