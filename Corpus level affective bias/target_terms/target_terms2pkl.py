#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 20 14:49:44 2022

@author: Anoop k
Target terms to pkl 
"""

import pickle

#data inputs 
basepath = 'emotion_terms/'
term_list_label = 'anger'
temp_term_list = []

#data pre-prcessing 
dupli_removed = list(dict.fromkeys(temp_term_list))
lowercasing = [x.lower() for x in dupli_removed]
lowercasing.sort()
term_list = lowercasing

#Save terms as as pkl file 
with open(str(basepath)+str(term_list_label)+'.pkl', 'wb') as f:
    pickle.dump(term_list, f)
print("sentence corpus saved")  


#Test read of pkl file for laterr use
with open(str(basepath)+str(term_list_label)+'.pkl', 'rb') as f:
    copy_var = pickle.load(f)

