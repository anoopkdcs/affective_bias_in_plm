# -*- coding: utf-8 -*-
"""
Created on Thu Apr 14 20:24:01 2022

@author: Anoop K

Affective Bias Finder 
    1. for sentences with ground truth
    2. along each PLMs 
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

#Senence pair reader function
def sent_pair_reader(file1_path,file2_path):
    
    # Read Sentence Paires 
    # file structure --> {Sentence, gtruth, prediction, Prediction Intensity}
    #file index --> {Sentence:0, gtruth:1, prediction:2, Prediction Intensity:3}
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
    
    # Plot Prediction Intensity 
    x_indices = []
    emo_intensity_list1 = []
    emo_intensity_list2 = []
    for i in range(len(data1)):
        if(data1[i][2] == data2[i][2]) & (data1[i][1] == data2[i][1]) & (data1[i][1] == data1[i][2]) & (data2[i][1] == data2[i][2]):
            x_indices.append(i)
            emo_intensity_list1.append(np.float64(data1[i][3]))
            emo_intensity_list2.append(np.float64(data2[i][3]))
            
            
    
    fig = plt.figure(figsize = (10, 5))
    plt.plot(x_indices,emo_intensity_list1, label = str(pair1_label) + " "+ str(model_name), color ='r', marker='o', linestyle = 'None') 
    plt.plot(x_indices,emo_intensity_list2, label = str(pair2_label) + " "+ str(model_name), color ='b', marker='o', linestyle = 'None') 
    
    plt.axhline(y = mean(emo_intensity_list1), color = 'r', linestyle = '--', label= str(pair1_label) + ' '+'mean')
    plt.axhline(y = mean(emo_intensity_list2), color = 'b', linestyle = '--', label= str(pair2_label) + ' '+'mean')
        
    plt.xlabel('Input Sentences', fontsize = 12)
    plt.ylabel('Predicted Emotion Intensity', fontsize = 12)
    plt.legend( loc='upper center', bbox_to_anchor=(.5, 1.15),
                  fancybox=True, shadow=True, ncol=4,fontsize=12 )
    plot_name = 'plots/'+str(context)+'/'+str(eval_corpus)+'/'+'intensity_plot_over_plm'+str(model_name)+'_'+str(pair1_label)+'_'+str(pair2_label)
    #plt.savefig(plot_name+'.png')
    plt.show()
    
    print("----" + str(model_name) + "----")
    
    #Mean Intensity Difference 
    print('Mean Intensity '+ str(pair1_label)+': ', mean(emo_intensity_list1))
    print('Mean Intensity '+ str(pair2_label)+': ', mean(emo_intensity_list2))
    print('Mean Intensity Difference: ', np.absolute(mean(emo_intensity_list1)-mean(emo_intensity_list2)))
    
    # Average delta  for each model_name
    delta_emotion = np.absolute(np.array(emo_intensity_list1) - np.array(emo_intensity_list2))
    #delta_emotion = np.absolute(np.float64(emo_intensity_matrix_pair1) - np.float64(emo_intensity_matrix_pair2))
    avg_delta = np.average(delta_emotion)
    print("average delta "+str(model_name)+ ": ",avg_delta)
    
    # Statistical significance over emotions
    value, pvalue = ttest_rel(emo_intensity_list1, emo_intensity_list2)#, equal_var=False
    print("significance test values for "+ str( pair1_label) + " and " + str(pair2_label)+ " with " + str(model_name)+": ", value, pvalue)
    print("\n")
    
    return
    
def confidence_score_finder(data1, data2, model_name):    
    confidence_scores = []
    for i in range(len(data1)):
        if (data1[i][2] == data2[i][2]) & (data1[i][1] == data2[i][1]) & (data1[i][1] == data1[i][2]) & (data2[i][1] == data2[i][2]):
            score_pair_1 = np.float64(data1[i][3])
            score_pair_2 = np.float64(data2[i][3])
            confidence = 1 - (score_pair_1/score_pair_2)
            confidence_scores.append(confidence)
            #if (model_name =='BERT'): #to veryfy outputs
                #print(score_pair_1)
                #print(score_pair_2)
    confidence_scores = np.array(confidence_scores)

    
    model_detail = np.full((confidence_scores.shape[0], 1), model_name)
    confidence_score_matrix = np.concatenate((np.float64(np.reshape(confidence_scores, 
                                                                    (len(confidence_scores),1))) ,
                                              model_detail), axis = 1)
    avg_cscore = np.average(confidence_scores)
    print("average Confident Score "+str(model_name)+ ": ",avg_cscore)
    return confidence_score_matrix



def confi_score_violine_plot(cscore1, cscore2, cscore3, cscore4, pair1, pair2):
    cs1_cs2 = np.concatenate((cscore1,cscore2), axis = 0)
    cs12_cs3 = np.concatenate((cs1_cs2,cscore3), axis = 0)
    cscore_all = np.concatenate((cs12_cs3,cscore4), axis = 0)
    
    # Violine plot for confidence scores 
    sns.set_style("whitegrid")
    sns.set_context("notebook")
    fig, ax = pyplot.subplots(dpi = 100)#figsize =(6, 3), 
    xx = sns.violinplot(ax = ax, x = cscore_all[:,1] , y = np.float64(cscore_all[:,0]),
                        hue = cscore_all[:,1], palette=sns.color_palette("husl",4))
    xx.set(xticklabels=[])
    plt.legend( loc='upper center', bbox_to_anchor=(.49, 1.15), fancybox=True, shadow=True, ncol=4,fontsize=12 )
    #plt.show()
    plt.xlabel(str(pair1)+' vs. '+str(pair2), fontsize = 12)
    plot_name = 'plots/'+str(context)+'/'+str(eval_corpus)+'/'+'confidence_plot_over_plm_'+str(context)+'_'+str(pair1_label)+'_'+str(pair2_label)
    #plt.savefig(plot_name+'.png')
    return


# --------- Main function ------
#Change Variables 
eval_corpus = 'bits'
context = 'race'
pair1_label = 'EA'
pair2_label = 'AA'
bert_pair1_path = 'sentence_paires_predictions/race/bits/bits_euro_american_120_bits_bert_predictions.csv'
bert_pair2_path = 'sentence_paires_predictions/race/bits/bits_afri_american_120_bits_bert_predictions.csv'

gpt2_pair1_path = 'sentence_paires_predictions/race/bits/bits_euro_american_120_bits_gpt2_predictions.csv'
gpt2_pair2_path = 'sentence_paires_predictions/race/bits/bits_afri_american_120_bits_gpt2_predictions.csv'

xlnet_pair1_path = 'sentence_paires_predictions/race/bits/bits_euro_american_120_bits_XLNet_predictions.csv'
xlnet_pair2_path = 'sentence_paires_predictions/race/bits/bits_afri_american_120_bits_XLNet_predictions.csv'

t5_pair1_path = 'sentence_paires_predictions/race/bits/bits_euro_american_120_bits_T5_predictions.csv'
t5_pair2_path = 'sentence_paires_predictions/race/bits/bits_afri_american_120_bits_T5_predictions.csv'

# Affective Bias finder
bert_pair_1, bert_pair_2 = sent_pair_reader(bert_pair1_path, bert_pair2_path)
gpt2_pair_1, gpt2_pair_2 = sent_pair_reader(gpt2_pair1_path, gpt2_pair2_path)
xlnet_pair_1, xlnet_pair_2 = sent_pair_reader(xlnet_pair1_path, xlnet_pair2_path)
t5_pair_1, t5_pair_2 = sent_pair_reader(t5_pair1_path, t5_pair2_path)

affective_bias_finder(bert_pair_1, bert_pair_2, pair1_label, pair2_label, 'BERT')
affective_bias_finder(gpt2_pair_1, gpt2_pair_2, pair1_label, pair2_label, 'GPT-2')
affective_bias_finder(xlnet_pair_1, xlnet_pair_2, pair1_label, pair2_label, 'XLNet')
affective_bias_finder(t5_pair_1, t5_pair_2, pair1_label, pair2_label, 'T5')

# confidence score plot
bert_confi_score = confidence_score_finder(bert_pair_1, bert_pair_2, 'BERT')
gpt2_confi_score = confidence_score_finder(gpt2_pair_1, gpt2_pair_2, 'GPT-2')
xlnet_confi_score = confidence_score_finder(xlnet_pair_1, xlnet_pair_2, 'XLNet')
t5_confi_score = confidence_score_finder(t5_pair_1, t5_pair_2, 'T5')

confi_score_violine_plot(bert_confi_score, gpt2_confi_score, xlnet_confi_score, t5_confi_score,
                         pair1_label, pair2_label)
