# -*- coding: utf-8 -*-
"""
Created on Sun Mar 13 12:53:44 2022

@author: Anoop K 
"""
import csv 

#read data
data_path = 'data/input.csv'
csvfile=open(data_path,'r')
obj=csv.reader(csvfile)
rows = []
for row in obj:
    rows.append(row)   
print(row)

#pre-processing
data = [x.lower() for x in row]
print("\n Normalised List")
print(data)

final_word_list = list(dict.fromkeys(data))
print("\n Duplication removed")
print(final_word_list)



