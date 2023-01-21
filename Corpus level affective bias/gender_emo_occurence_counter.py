#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 18 11:48:30 2022

@author: Anoop K
https://github.com/tanyichern/social-biases-contextualized
Corpus level affective bias in Gender: co-occurrence of gender emotion terms 
"""
from collections import Counter
from multiprocessing import Pool
import datetime
import sys

# global variables 
global anger_wdrs
global fear_wrds
global joy_wrds
global sadness_wrds
global m_tgts
global f_tgts
global nb_tgts
    

def update_count_para(sentence):
    counts = {"anger_count": 0,
              "fear_count": 0,
              "joy_count": 0,
              "sadness_count": 0,
              "anger_male": 0,
              "fear_male": 0,
              "joy_male": 0,
              "sadness_male": 0,
              "anger_female": 0,
              "fear_female": 0,
              "joy_female": 0,
              "sadness_female":0,
              "anger_non_binary": 0,
              "fear_non_binary": 0,
              "joy_non_binary": 0,
              "sadness_non_binary": 0 }
    
    counts = Counter(counts)
    tokens = sentence.lower().split()
    anger_occ = False
    fear_occ = False
    joy_occ = False
    sadness_occ = False

    for anger_wdr in anger_wdrs:
        if anger_wdr in tokens:
            anger_occ = True
            counts["anger_count"] += 1
            break

    for fear_wrd in fear_wrds:
        if fear_wrd in tokens:
            fear_occ = True
            counts["fear_count"] += 1
            break

    for joy_wrd in joy_wrds:
        if joy_wrd in tokens:
            joy_occ = True
            counts["joy_count"] += 1
            break

    for sadness_wrd in sadness_wrds:
        if sadness_wrd in tokens:
            sadness_occ = True
            counts["sadness_count"] += 1
            break
#for male
    for m_tgt in m_tgts:
        if m_tgt in tokens and anger_occ:
            counts["anger_male"] += 1
            break
        if m_tgt in tokens and fear_occ:
            counts["fear_male"] += 1
            break
        if m_tgt in tokens and joy_occ:
            counts["joy_male"] += 1
            break
        if m_tgt in tokens and sadness_occ:
            counts["sadness_male"] +=1
            break

#for female
    for f_tgt in f_tgts:
        if f_tgt in tokens and anger_occ:
            counts["anger_female"] += 1
            break
        if f_tgt in tokens and fear_occ:
            counts["fear_female"] += 1
            break
        if f_tgt in tokens and joy_occ:
            counts["joy_female"] += 1
            break
        if f_tgt in tokens and sadness_occ:
            counts["sadness_female"] += 1
            break

#for non-binary          
    for nb_tgt in nb_tgts:
        if nb_tgt in tokens and anger_occ:
            counts["anger_non_binary"] += 1
            break
        if nb_tgt in tokens and fear_occ:
            counts["fear_non_binary"] += 1
            break
        if nb_tgt in tokens and joy_occ:
            counts["joy_non_binary"] += 1
            break
        if nb_tgt in tokens and sadness_occ:
            counts["sadness_non_binary"] += 1
            break

    return counts

sentence = ["he him anger irritability",
            "she sadness",
            "his joy",
            "he is anger about homosexual "]

#Gender target words 
m_tgts = ["he", "him", "his"]
f_tgts = ["she", "her", "hers"]
nb_tgts = ["bisexual", "homosexual"]

anger_wdrs = ["anger","irritability"]
fear_wrds = ["fear", "horror"]
joy_wrds = ["joy", "cheerfulness"]
sadness_wrds = ["sadness","suffering"]



workers = 16
with Pool(processes=workers) as pool:
    results = pool.map(update_count_para, sentence)
    print(f"finished count, {datetime.datetime.now()}")
    sys.stdout.flush()

    # accumulate counts
    print(f"starting acc, {datetime.datetime.now()}")
    sys.stdout.flush()
    master_count = sum(results, Counter())
    print(f"finished acc, {datetime.datetime.now()}")
    sys.stdout.flush()
    
    # print counts
    for key, value in master_count.items():
        print(f"{key}: {value}")

