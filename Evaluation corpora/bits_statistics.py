#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 21 16:12:40 2022

@author: user
"""
import csv
import numpy as np 

#### BITS Gender data ####
gender_file = open('data/bits_gender_anger_fear_joy_sad.csv')
gender_csvreader = csv.reader(gender_file)
gedner_header = next(gender_csvreader)
gender_rows = []

for i in gender_csvreader:
        gender_rows.append(i)
gender_rows = np.array(gender_rows) 


#### BITS Race data ####
race_file = open('data/bits_race_anger_fear_joy_sad.csv')
race_csvreader = csv.reader(race_file)
race_header = next(race_csvreader)
race_rows = []

for i in race_csvreader:
        race_rows.append(i)
race_rows = np.array(race_rows) 


#statistics
total_sent =  len(gender_rows) + len(race_rows)

#Gender
gender_cat = np.unique(gender_rows[:,3])
total_male =  len(np.where(gender_rows[:,3]=='Male')[0])
total_female =  len(np.where(gender_rows[:,3]=='Female')[0])
total_non_binary =  len(np.where(gender_rows[:,3]=='Non-Binary')[0])


#Race
race_cat = np.unique(race_rows[:,3])
total_american_indian=  len(np.where(race_rows[:,3]=='American Indian')[0])
total_asian =  len(np.where(race_rows[:,3]=='Asian')[0])
total_black =  len(np.where(race_rows[:,3]=='Black')[0])
total_latino =  len(np.where(race_rows[:,3]=='Latino')[0])
total_white =  len(np.where(race_rows[:,3]=='White')[0])




#statistics print 
print("\ntotal number of sentences: ", total_sent)

print("\n------Gender------- ")
print("Gender categories: ", gender_cat)
print("total number of males: ", total_male)
print("total number of females: ", total_female)
print("total number of non-binary: ", total_non_binary)

print("\n------Race------- ")
print("Race categories: ", race_cat)
print("total number of American Indian: ", total_american_indian)
print("total number of Asian: ", total_asian)
print("total number of Black: ", total_black)
print("total number of Latino: ", total_latino)
print("total number of White: ", total_white)