#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 21 15:14:31 2022

@author: user
"""

import csv
import numpy as np 

#### Read EEC  ####
eec_file = open('data/eec.csv')
eec_csvreader = csv.reader(eec_file)
eec_header = next(eec_csvreader)
eec_rows = []

for i in eec_csvreader:
        eec_rows.append(i)
eec_rows = np.array(eec_rows) 

#statistics
total_sent =  len(eec_rows)

#Gender
gender_cat = np.unique(eec_rows[:,4])
total_male =  len(np.where(eec_rows[:,4]=='male')[0])
total_female =  len(np.where(eec_rows[:,4]=='female')[0])

#Race
race_cat = np.unique(eec_rows[:,5])
total_african=  len(np.where(eec_rows[:,5]=='African+AC0-American')[0])
total_european =  len(np.where(eec_rows[:,5]=='European')[0])

#intersectional [Race+Gender]
total_male_african = len(np.where((eec_rows[:,4]=='male') & (eec_rows[:,5] == 'African+AC0-American'))[0])
total_male_european= len(np.where((eec_rows[:,4]=='male') & (eec_rows[:,5] == 'European'))[0])
total_male_alone= len(np.where((eec_rows[:,4]=='male') & (eec_rows[:,5] == ''))[0])

total_female_african = len(np.where((eec_rows[:,4]=='female') & (eec_rows[:,5] == 'African+AC0-American'))[0])
total_female_european= len(np.where((eec_rows[:,4]=='female') & (eec_rows[:,5] == 'European'))[0])
total_female_alone= len(np.where((eec_rows[:,4]=='female') & (eec_rows[:,5] == ''))[0])


#statistics print 
print("\ntotal number of sentences: ", total_sent)




print("\n------Gender------- ")
print("Gender categories: ", gender_cat)
print("total number of males: ", total_male)
print("total number of females: ", total_female)

print("\n------Race------- ")
print("Race categories: ", race_cat)
print("total number of African: ", total_african)
print("total number of European: ", total_european)

print("\n------Intersectional------- ")
print("total male african: ", total_male_african)
print("total male european: ", total_male_european)
print("total male alone: ", total_male_alone)

print("total female african: ", total_female_african)
print("total female european: ", total_female_european)
print("total female alone: ", total_female_alone)

