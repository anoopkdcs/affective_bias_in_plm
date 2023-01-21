# -*- coding: utf-8 -*-
"""
Created on Fri Apr  1 17:29:57 2022

@author: Anoop K. 
CrowS Pair Evaluation corpus reading, preprocessing and statistics 
pre-processing :- only select gender, race, and religion oriented text instances 
"""

import csv
import numpy as np 
import pandas as pd  

data_path = open('data/csp/CrowS-Pairs.csv') 
csp_csvreader = csv.reader(data_path)
csp_header = next(csp_csvreader)
csp_rows = []

for i in csp_csvreader:
        csp_rows.append(i)
        
csp_rows = np.array(csp_rows) 

#remove un wanted coloumns 
csp_cols_removed = csp_rows[:,1:5]

csp_gender = []
csp_race = []
csp_religion = []

for i in range(len(csp_cols_removed)):
    tmp_bias_type = csp_cols_removed[i][3]
    tmp_data = csp_cols_removed[i]
    
    if tmp_bias_type == 'gender':
        csp_gender.append(tmp_data) #262 instances
        
    if tmp_bias_type == 'race-color':
        csp_race.append(tmp_data) #516 instances
        
    if tmp_bias_type == 'religion':
        csp_religion.append(tmp_data) #105 instances 

csp_gender = np.array(csp_gender)
csp_race = np.array(csp_race)
csp_religion = np.array(csp_religion)


csp_gender_pd = pd.DataFrame(csp_gender)
csp_gender_pd.to_csv("data/csp/CrowS-Pairs_gender.csv")

csp_race_pd = pd.DataFrame(csp_race)
csp_race_pd.to_csv("data/csp/CrowS-Pairs_race.csv")

csp_religion_pd = pd.DataFrame(csp_religion)
csp_religion_pd.to_csv("data/csp/CrowS-Pairs_religion.csv")
