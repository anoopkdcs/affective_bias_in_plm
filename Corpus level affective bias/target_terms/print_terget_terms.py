# -*- coding: utf-8 -*-
"""
Created on Tue May 10 20:06:38 2022

@author: print target terms
"""
import pickle

#data inputs 
basepath = 'religion_terms/'
term_list_label = 'christian'
#Test read of pkl file for laterr use
with open(str(basepath)+str(term_list_label)+'.pkl', 'rb') as f:
    copy_var = pickle.load(f)

print(copy_var)
